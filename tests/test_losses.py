from __future__ import annotations

import copy
import json
import random

import pytest
import torch
import torch.nn.functional as F

from microtrain.dpo import dpo_loss
from microtrain.environment import Task
from microtrain.grpo import (
    GRPOConfig,
    NoValidToolCallsError,
    balanced_prompt_sample,
    check_tool_call_guard,
    grpo_loss,
    group_normalize,
    train_grpo,
    write_representative_trajectories,
)
from microtrain.model import ModelConfig, SmolLM
from microtrain.rewards import RewardBreakdown
from microtrain.rollout import (
    TrainingSequence,
    collate_sequences,
    conversation_sequence,
    prompt_response_sequence,
)

from .helpers import CharTokenizer


def test_completion_mask_excludes_prompt_and_includes_eos() -> None:
    tokenizer = CharTokenizer()
    sequence = prompt_response_sequence(tokenizer, "prompt", "answer")
    _, _, mask = collate_sequences([sequence], tokenizer.pad_id, "cpu")
    assert int(mask.sum()) == len("answer") + 1
    assert not bool(mask[0, : len("prompt")].any())


def test_conversation_mask_excludes_tool_observation() -> None:
    tokenizer = CharTokenizer()
    first = " <tool>{}</tool>"
    observation = "\nTool: <result>3</result>\nAssistant:"
    final = " <answer>3</answer>"
    sequence = conversation_sequence(tokenizer, "prompt", first + observation + final)
    _, _, mask = collate_sequences([sequence], tokenizer.pad_id, "cpu")
    assert int(mask.sum()) == len(first) + len(final) + 1


def test_dpo_loss_matches_direct_formula_and_detaches_reference() -> None:
    policy_chosen = torch.tensor([2.0], requires_grad=True)
    policy_rejected = torch.tensor([1.0], requires_grad=True)
    reference_chosen = torch.tensor([1.5], requires_grad=True)
    reference_rejected = torch.tensor([1.0], requires_grad=True)
    loss, margin = dpo_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.2
    )
    expected_margin = torch.tensor([0.1])
    torch.testing.assert_close(margin, expected_margin)
    torch.testing.assert_close(loss, -F.logsigmoid(expected_margin).mean())
    loss.backward()
    assert policy_chosen.grad is not None
    assert reference_chosen.grad is None


def test_group_normalization_and_zero_variance() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    advantages = group_normalize(rewards)
    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        advantages[0].std(unbiased=False), torch.tensor(1.0), atol=1e-6, rtol=0
    )
    assert torch.equal(advantages[1], torch.zeros(3))


def test_grpo_masks_padding_and_detaches_old_policy() -> None:
    new = torch.zeros((2, 3), requires_grad=True)
    old = torch.zeros((2, 3), requires_grad=True)
    reference = torch.zeros((2, 3), requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    loss, diagnostics = grpo_loss(
        new,
        old,
        reference,
        mask,
        advantages,
        clip_epsilon=0.2,
        kl_beta=0.02,
    )
    loss.backward()
    assert new.grad is not None
    assert new.grad[0, 2] == 0
    assert old.grad is None
    assert reference.grad is None
    assert advantages.grad is None
    assert diagnostics["kl"] == 0


def test_grpo_default_reuses_rollouts_so_clipping_can_activate() -> None:
    assert GRPOConfig().update_epochs > 1


def test_balanced_prompt_sample_includes_every_difficulty() -> None:
    tasks = [
        Task(f"{difficulty}-{index}", "1 + 1", 2, difficulty, "test")
        for difficulty in ("easy", "medium", "hard")
        for index in range(3)
    ]
    selected = balanced_prompt_sample(tasks, 6, random.Random(7))
    counts = {
        difficulty: sum(task.difficulty == difficulty for task in selected)
        for difficulty in ("easy", "medium", "hard")
    }
    assert counts == {"easy": 2, "medium": 2, "hard": 2}


def test_tool_call_guard_stops_after_initial_updates_without_execution() -> None:
    check_tool_call_guard(
        tools=True,
        step=4,
        total_steps=50,
        guard_steps=5,
        valid_tool_calls=0,
    )
    with pytest.raises(NoValidToolCallsError, match="first 5 updates"):
        check_tool_call_guard(
            tools=True,
            step=5,
            total_steps=50,
            guard_steps=5,
            valid_tool_calls=0,
        )
    check_tool_call_guard(
        tools=True,
        step=5,
        total_steps=50,
        guard_steps=5,
        valid_tool_calls=1,
    )


def test_representative_training_trajectories_are_saved(tmp_path) -> None:
    task = Task("hard-1", "20 * 20", 400, "hard", "test")
    sequences = [
        TrainingSequence(
            prompt=task.prompt,
            token_ids=[0, 1],
            policy_mask=[False, True],
            reward=reward,
            response=response,
            breakdown=RewardBreakdown(
                total=reward,
                correct=correct,
                valid_format=correct,
                used_tool=correct,
                attempted_tool=True,
            ),
        )
        for reward, response, correct in [
            (-0.25, " <tool>-1</tool>", False),
            (0.88, ' <tool>{"expression":"20 * 20"}</tool>', True),
        ]
    ]
    path = tmp_path / "trajectories.jsonl"
    write_representative_trajectories(
        path,
        step=3,
        tasks=[task],
        sequences=sequences,
        group_size=2,
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["representative"] for record in records] == ["lowest", "highest"]
    assert records[0]["attempted_tool"] and not records[0]["executed_tool"]
    assert records[1]["executed_tool"]


def test_grpo_accumulates_one_update_across_balanced_prompts(tmp_path) -> None:
    tokenizer = CharTokenizer()
    model = SmolLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    )
    reference = copy.deepcopy(model)
    tasks = [
        Task(f"{difficulty}-1", "1 + 1", 2, difficulty, "test")
        for difficulty in ("easy", "medium", "hard")
    ]
    metrics = train_grpo(
        model,
        reference,
        tokenizer,  # type: ignore[arg-type]
        tasks,
        GRPOConfig(
            steps=1,
            prompts_per_step=3,
            group_size=2,
            update_epochs=1,
            max_new_tokens=2,
            temperature=1.0,
        ),
        "cpu",
        balanced_difficulties=True,
        trajectory_output=tmp_path / "trajectories.jsonl",
    )
    assert len(metrics) == 1
    assert len((tmp_path / "trajectories.jsonl").read_text().splitlines()) == 6


def test_agent_grpo_stops_and_preserves_evidence_without_valid_tools(
    tmp_path, monkeypatch
) -> None:
    tokenizer = CharTokenizer()
    model = SmolLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    )
    reference = copy.deepcopy(model)
    tasks = [
        Task(f"{difficulty}-1", "1 + 1", 2, difficulty, "test")
        for difficulty in ("easy", "medium", "hard")
    ]

    def no_tool_groups(model, reference, tokenizer, tasks, config, device, *, tools):
        del model, reference, tokenizer, device, tools
        sequences = [
            TrainingSequence(
                prompt=task.prompt,
                token_ids=[0, 1],
                policy_mask=[False, True],
                old_logprobs=[0.0],
                reference_logprobs=[0.0],
                reward=0.1,
                response=" <answer>3</answer>",
                breakdown=RewardBreakdown(
                    total=0.1,
                    correct=False,
                    valid_format=True,
                    used_tool=False,
                    attempted_tool=False,
                ),
            )
            for task in tasks
            for _ in range(config.group_size)
        ]
        return sequences, torch.zeros(len(sequences))

    monkeypatch.setattr("microtrain.grpo.collect_groups", no_tool_groups)
    logged: list[dict[str, float | int]] = []
    path = tmp_path / "guarded-trajectories.jsonl"
    with pytest.raises(NoValidToolCallsError, match="first 5 updates") as caught:
        train_grpo(
            model,
            reference,
            tokenizer,  # type: ignore[arg-type]
            tasks,
            GRPOConfig(
                steps=6,
                prompts_per_step=3,
                group_size=2,
                update_epochs=1,
                max_new_tokens=2,
                temperature=0.9,
                learning_rate=0.0,
                tool_guard_steps=5,
            ),
            "cpu",
            tools=True,
            balanced_difficulties=True,
            trajectory_output=path,
            on_log=logged.append,
        )
    assert len(logged) == 5
    assert len(caught.value.metrics) == 5
    assert len(path.read_text().splitlines()) == 5 * 3 * 2
