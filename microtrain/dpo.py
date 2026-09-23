"""Direct preference optimization with a frozen SFT reference policy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .model import SmolLM
from .rollout import TrainingSequence, conversation_sequence, sequence_logprobs
from .tokenizer import MicroTokenizer


@dataclass(frozen=True)
class DPOConfig:
    steps: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-6
    beta: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 10


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_chosen = reference_chosen.detach()
    reference_rejected = reference_rejected.detach()
    policy_ratio = policy_chosen - policy_rejected
    reference_ratio = reference_chosen - reference_rejected
    margin = beta * (policy_ratio - reference_ratio)
    return -F.logsigmoid(margin).mean(), margin.detach()


@torch.inference_mode()
def precompute_reference_logprobs(
    reference: SmolLM,
    chosen: Sequence[TrainingSequence],
    rejected: Sequence[TrainingSequence],
    tokenizer: MicroTokenizer,
    device: str | torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference.eval()
    chosen_values: list[torch.Tensor] = []
    rejected_values: list[torch.Tensor] = []
    for start in range(0, len(chosen), batch_size):
        chosen_values.append(
            sequence_logprobs(reference, chosen[start : start + batch_size], tokenizer.pad_id, device).cpu()
        )
        rejected_values.append(
            sequence_logprobs(reference, rejected[start : start + batch_size], tokenizer.pad_id, device).cpu()
        )
    return torch.cat(chosen_values), torch.cat(rejected_values)


def train_dpo(
    model: SmolLM,
    reference: SmolLM,
    tokenizer: MicroTokenizer,
    examples: Sequence[dict[str, str]],
    config: DPOConfig,
    device: str | torch.device,
    *,
    on_log: Callable[[dict[str, float | int]], None] | None = None,
) -> list[dict[str, float | int]]:
    if not examples:
        raise ValueError("DPO requires at least one preference pair")
    chosen = [
        conversation_sequence(tokenizer, example["prompt"], example["chosen"])
        for example in examples
    ]
    rejected = [
        conversation_sequence(tokenizer, example["prompt"], example["rejected"])
        for example in examples
    ]
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    ref_chosen, ref_rejected = precompute_reference_logprobs(
        reference, chosen, rejected, tokenizer, device, config.batch_size
    )

    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    metrics: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, config.steps + 1):
        indices = [rng.randrange(len(examples)) for _ in range(config.batch_size)]
        chosen_batch = [chosen[index] for index in indices]
        rejected_batch = [rejected[index] for index in indices]
        policy_values = sequence_logprobs(
            model, chosen_batch + rejected_batch, tokenizer.pad_id, device
        )
        policy_chosen, policy_rejected = policy_values.chunk(2)
        reference_chosen = ref_chosen[indices].to(device)
        reference_rejected = ref_rejected[indices].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, margins = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            config.beta,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "margin": float(margins.mean()),
            "preference_accuracy": float((margins > 0).float().mean()),
            "grad_norm": float(grad_norm),
        }
        metrics.append(record)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            if on_log is not None:
                on_log(record)
    return metrics
