from __future__ import annotations

import torch
import torch.nn.functional as F

from microtrain.dpo import dpo_loss
from microtrain.grpo import GRPOConfig, grpo_loss, group_normalize
from microtrain.rollout import collate_sequences, conversation_sequence, prompt_response_sequence

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
