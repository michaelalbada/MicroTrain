"""Supervised fine-tuning: completion-only cross-entropy in one explicit loop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .model import SmolLM
from .rollout import TrainingSequence, collate_sequences, conversation_sequence
from .tokenizer import MicroTokenizer


@dataclass(frozen=True)
class SFTConfig:
    steps: int = 200
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 10


def sft_loss(
    model: SmolLM,
    sequences: Sequence[TrainingSequence],
    pad_id: int,
    device: str | torch.device,
) -> torch.Tensor:
    inputs, targets, completion_mask = collate_sequences(sequences, pad_id, device)
    logits = model(inputs)
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    return (token_losses * completion_mask).sum() / completion_mask.sum().clamp_min(1)


def train_sft(
    model: SmolLM,
    tokenizer: MicroTokenizer,
    examples: Sequence[dict[str, str]],
    config: SFTConfig,
    device: str | torch.device,
) -> list[dict[str, float | int]]:
    if not examples:
        raise ValueError("SFT requires at least one example")
    sequences = [
        conversation_sequence(tokenizer, example["prompt"], example["completion"])
        for example in examples
    ]
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    metrics: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, config.steps + 1):
        batch = [sequences[rng.randrange(len(sequences))] for _ in range(config.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss = sft_loss(model, batch, tokenizer.pad_id, device)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            metrics.append(
                {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm)}
            )
    return metrics
