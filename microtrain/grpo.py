"""Group-relative policy optimization for responses and tool-use episodes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from .environment import Task
from .model import SmolLM
from .rollout import (
    TrainingSequence,
    attach_reference_logprobs,
    collate_sequences,
    sample_episode,
    sample_group,
    token_logprobs,
)
from .tokenizer import MicroTokenizer


@dataclass(frozen=True)
class GRPOConfig:
    steps: int = 50
    prompts_per_step: int = 1
    group_size: int = 8
    update_epochs: int = 2
    max_new_tokens: int = 32
    max_turns: int = 2
    learning_rate: float = 1e-6
    temperature: float = 1.2
    clip_epsilon: float = 0.2
    kl_beta: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 1
    tool_guard_steps: int = 5


class NoValidToolCallsError(RuntimeError):
    """Raised before Agent RL can persist a policy that never executes its tool."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.metrics: list[dict[str, float | int]] = []


def balanced_prompt_sample(
    tasks: Sequence[Task],
    count: int,
    rng: random.Random,
) -> list[Task]:
    """Sample a per-update batch with near-equal difficulty representation."""

    difficulties = ("easy", "medium", "hard")
    if count < len(difficulties):
        raise ValueError("balanced prompt batches require at least three prompts")
    buckets = {
        difficulty: [task for task in tasks if task.difficulty == difficulty]
        for difficulty in difficulties
    }
    missing = [difficulty for difficulty, bucket in buckets.items() if not bucket]
    if missing:
        raise ValueError(f"balanced prompt batch is missing difficulties: {missing}")

    base, remainder = divmod(count, len(difficulties))
    extra_order = list(difficulties)
    rng.shuffle(extra_order)
    quotas = {difficulty: base for difficulty in difficulties}
    for difficulty in extra_order[:remainder]:
        quotas[difficulty] += 1
    selected = [
        rng.choice(buckets[difficulty])
        for difficulty in difficulties
        for _ in range(quotas[difficulty])
    ]
    rng.shuffle(selected)
    return selected


def check_tool_call_guard(
    *,
    tools: bool,
    step: int,
    total_steps: int,
    guard_steps: int,
    valid_tool_calls: int,
) -> None:
    if (
        tools
        and guard_steps > 0
        and total_steps >= guard_steps
        and step == guard_steps
        and valid_tool_calls == 0
    ):
        raise NoValidToolCallsError(
            f"no valid calculator call was executed in the first {guard_steps} updates; "
            "inspect the saved Agent-RL trajectories before retrying"
        )


def write_representative_trajectories(
    path: Path,
    *,
    step: int,
    tasks: Sequence[Task],
    sequences: Sequence[TrainingSequence],
    group_size: int,
) -> None:
    """Append the lowest/highest-reward trajectory for every prompt group."""

    with path.open("a") as handle:
        for prompt_index, task in enumerate(tasks):
            start = prompt_index * group_size
            group = sequences[start : start + group_size]
            ranked = sorted(enumerate(group), key=lambda item: item[1].reward)
            representatives = (("lowest", ranked[0]), ("highest", ranked[-1]))
            for representative, (sample_index, sequence) in representatives:
                breakdown = sequence.breakdown
                record = {
                    "kind": "training_trajectory",
                    "step": step,
                    "prompt_id": task.id,
                    "difficulty": task.difficulty,
                    "expression": task.expression,
                    "representative": representative,
                    "sample_index": sample_index,
                    "response": sequence.response,
                    "reward": sequence.reward,
                    "correct": bool(breakdown and breakdown.correct),
                    "valid_format": bool(breakdown and breakdown.valid_format),
                    "attempted_tool": bool(breakdown and breakdown.attempted_tool),
                    "executed_tool": bool(breakdown and breakdown.used_tool),
                    "policy_tokens": sum(sequence.policy_mask),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def group_normalize(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each prompt's group; constant-reward groups produce zero advantage."""

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [prompts, group]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, keepdim=True, unbiased=False)
    return torch.where(scale > eps, centered / scale.clamp_min(eps), torch.zeros_like(centered))


def _padded_values(
    sequences: Sequence[TrainingSequence],
    values: str,
    token_mask: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    padded = torch.zeros_like(token_mask, dtype=torch.float32, device=device)
    for row, sequence in enumerate(sequences):
        sequence_values = getattr(sequence, values)
        if len(sequence_values) != int(token_mask[row].sum()):
            raise ValueError(f"{values} does not align with generated-token mask")
        padded[row][token_mask[row]] = torch.tensor(sequence_values, device=device)
    return padded


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    old_logprobs = old_logprobs.detach()
    reference_logprobs = reference_logprobs.detach()
    advantages = advantages.detach()
    ratio = torch.exp(new_logprobs - old_logprobs)
    advantage = advantages.unsqueeze(1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    policy_loss = -torch.minimum(unclipped, clipped)

    # Positive sampled KL estimator used by GRPO implementations.
    log_ratio = reference_logprobs - new_logprobs
    sampled_kl = (torch.exp(log_ratio) - log_ratio - 1.0).clamp_min(0.0)
    denominator = token_mask.sum().clamp_min(1)
    policy_mean = (policy_loss * token_mask).sum() / denominator
    kl_mean = (sampled_kl * token_mask).sum() / denominator
    loss = policy_mean + kl_beta * kl_mean
    diagnostics = {
        "policy_loss": policy_mean.detach(),
        "kl": kl_mean.detach(),
        "clip_fraction": (((ratio - 1.0).abs() > clip_epsilon) * token_mask).sum().float()
        / denominator,
    }
    return loss, diagnostics


def collect_groups(
    model: SmolLM,
    reference: SmolLM,
    tokenizer: MicroTokenizer,
    tasks: Sequence[Task],
    config: GRPOConfig,
    device: str | torch.device,
    *,
    tools: bool,
) -> tuple[list[TrainingSequence], torch.Tensor]:
    groups: list[list[TrainingSequence]] = []
    model.eval()
    for task in tasks:
        if tools:
            group = [
                sample_episode(
                    model,
                    tokenizer,
                    task,
                    max_turns=config.max_turns,
                    max_action_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    device=device,
                )
                for _ in range(config.group_size)
            ]
        else:
            group = sample_group(
                model,
                tokenizer,
                task,
                group_size=config.group_size,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                device=device,
            )
        groups.append(group)
    flattened = [sample for group in groups for sample in group]
    attach_reference_logprobs(
        reference,
        flattened,
        tokenizer.pad_id,
        device,
        temperature=config.temperature,
    )
    rewards = torch.tensor(
        [[sample.reward for sample in group] for group in groups],
        dtype=torch.float32,
        device=device,
    )
    return flattened, group_normalize(rewards).flatten()


def train_grpo(
    model: SmolLM,
    reference: SmolLM,
    tokenizer: MicroTokenizer,
    tasks: Sequence[Task],
    config: GRPOConfig,
    device: str | torch.device,
    *,
    tools: bool = False,
    balanced_difficulties: bool = False,
    trajectory_output: str | Path | None = None,
    on_log: Callable[[dict[str, float | int]], None] | None = None,
) -> list[dict[str, float | int]]:
    if not tasks:
        raise ValueError("GRPO requires at least one task")
    if config.temperature <= 0:
        raise ValueError("GRPO sampling temperature must be positive")
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference.eval()
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    metrics: list[dict[str, float | int]] = []
    valid_tool_calls = 0
    trajectory_path = Path(trajectory_output) if trajectory_output is not None else None
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text("")

    for step in range(1, config.steps + 1):
        prompt_tasks = (
            balanced_prompt_sample(tasks, config.prompts_per_step, rng)
            if balanced_difficulties
            else [tasks[rng.randrange(len(tasks))] for _ in range(config.prompts_per_step)]
        )
        sequences, advantages = collect_groups(
            model, reference, tokenizer, prompt_tasks, config, device, tools=tools
        )
        if trajectory_path is not None:
            write_representative_trajectories(
                trajectory_path,
                step=step,
                tasks=prompt_tasks,
                sequences=sequences,
                group_size=config.group_size,
            )
        reward_mean = sum(sequence.reward for sequence in sequences) / len(sequences)
        correct_rate = sum(
            bool(sequence.breakdown and sequence.breakdown.correct) for sequence in sequences
        ) / len(sequences)
        valid_rate = sum(
            bool(sequence.breakdown and sequence.breakdown.valid_format) for sequence in sequences
        ) / len(sequences)
        tool_attempts = sum(
            bool(sequence.breakdown and sequence.breakdown.attempted_tool)
            for sequence in sequences
        )
        executed_tool_calls = sum(
            bool(sequence.breakdown and sequence.breakdown.used_tool) for sequence in sequences
        )
        tool_attempt_rate = tool_attempts / len(sequences)
        tool_use_rate = executed_tool_calls / len(sequences)
        valid_tool_call_rate = (
            executed_tool_calls / tool_attempts if tool_attempts else 0.0
        )
        valid_tool_calls += executed_tool_calls
        mean_length = sum(sum(sequence.policy_mask) for sequence in sequences) / len(sequences)
        entropy_values = [value for sequence in sequences for value in sequence.entropies]
        mean_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
        zero_variance = float(
            advantages.view(config.prompts_per_step, config.group_size)
            .abs()
            .sum(dim=1)
            .eq(0)
            .float()
            .mean()
        )

        prompt_count = len(prompt_tasks)
        for _ in range(config.update_epochs):
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            diagnostic_values: dict[str, list[torch.Tensor]] = {
                "policy_loss": [],
                "kl": [],
                "clip_fraction": [],
            }
            for prompt_index in range(prompt_count):
                start = prompt_index * config.group_size
                stop = start + config.group_size
                prompt_sequences = sequences[start:stop]
                prompt_advantages = advantages[start:stop]
                inputs, targets, token_mask = collate_sequences(
                    prompt_sequences, tokenizer.pad_id, device
                )
                old_logprobs = _padded_values(
                    prompt_sequences, "old_logprobs", token_mask, device
                )
                reference_logprobs = _padded_values(
                    prompt_sequences, "reference_logprobs", token_mask, device
                )
                model.train()
                new_logprobs = token_logprobs(model, inputs, targets, config.temperature)
                prompt_loss, prompt_diagnostics = grpo_loss(
                    new_logprobs,
                    old_logprobs,
                    reference_logprobs,
                    token_mask,
                    prompt_advantages,
                    clip_epsilon=config.clip_epsilon,
                    kl_beta=config.kl_beta,
                )
                (prompt_loss / prompt_count).backward()
                losses.append(prompt_loss.detach())
                for name, value in prompt_diagnostics.items():
                    diagnostic_values[name].append(value)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            loss = torch.stack(losses).mean()
            diagnostics = {
                name: torch.stack(values).mean()
                for name, values in diagnostic_values.items()
            }

        record = {
            "step": step,
            "loss": float(loss.detach()),
            "reward": reward_mean,
            "correct_rate": correct_rate,
            "valid_rate": valid_rate,
            "tool_attempt_rate": tool_attempt_rate,
            "valid_tool_call_rate": valid_tool_call_rate,
            "tool_use_rate": tool_use_rate,
            "mean_length": mean_length,
            "entropy": mean_entropy,
            "kl": float(diagnostics["kl"]),
            "clip_fraction": float(diagnostics["clip_fraction"]),
            "zero_variance_groups": zero_variance,
            "grad_norm": float(grad_norm),
        }
        metrics.append(record)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            if on_log is not None:
                on_log(record)
        try:
            check_tool_call_guard(
                tools=tools,
                step=step,
                total_steps=config.steps,
                guard_steps=config.tool_guard_steps,
                valid_tool_calls=valid_tool_calls,
            )
        except NoValidToolCallsError as error:
            error.metrics = list(metrics)
            raise
    return metrics
