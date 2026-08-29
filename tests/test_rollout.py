from __future__ import annotations

import torch

from microtrain.rollout import TrainingSequence, collate_sequences


def test_episode_mask_only_marks_assistant_generated_tokens() -> None:
    sequence = TrainingSequence(
        prompt="p",
        token_ids=[0, 10, 11, 20, 21, 30, 31, 40, 41],
        policy_mask=[False, False, False, True, True, False, False, True, True],
        old_logprobs=[-1.0, -1.1, -0.9, -0.8],
    )
    _, _, mask = collate_sequences([sequence], pad_id=0, device="cpu")
    expected = torch.tensor([[False, False, True, True, False, False, True, True]])
    assert torch.equal(mask, expected)


def test_training_sequence_rejects_misaligned_old_logprobs() -> None:
    try:
        TrainingSequence(
            prompt="p",
            token_ids=[0, 1, 2],
            policy_mask=[False, True, True],
            old_logprobs=[-1.0],
        )
    except ValueError as error:
        assert "one value per generated token" in str(error)
    else:
        raise AssertionError("misaligned old log probabilities were accepted")

