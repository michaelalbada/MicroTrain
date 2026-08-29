"""One evaluation suite shared by the base and every post-training stage."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

import torch

from .data import preference_example
from .environment import Task
from .model import SmolLM
from .rollout import (
    TrainingSequence,
    attach_reference_logprobs,
    conversation_sequence,
    sample_episode,
    sample_group,
    sequence_logprobs,
)
from .tokenizer import MicroTokenizer


@dataclass(frozen=True)
class EvalConfig:
    max_new_tokens: int = 32
    max_turns: int = 2
    preference_batch_size: int = 16
    bootstrap_samples: int = 500
    seed: int = 17


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean(values: Sequence[float | int]) -> float:
    return mean(values) if values else 0.0


def bootstrap_interval(
    values: Sequence[bool], *, samples: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = sorted(
        _rate([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low = estimates[max(0, math.floor(tail * samples))]
    high = estimates[min(samples - 1, math.ceil((1.0 - tail) * samples) - 1)]
    return low, high


@torch.inference_mode()
def heldout_preference_metrics(
    model: SmolLM,
    tokenizer: MicroTokenizer,
    tasks: Sequence[Task],
    device: str | torch.device,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    pairs = [preference_example(task, rng, index) for index, task in enumerate(tasks)]
    chosen = [
        conversation_sequence(tokenizer, pair["prompt"], pair["chosen"])
        for pair in pairs
    ]
    rejected = [
        conversation_sequence(tokenizer, pair["prompt"], pair["rejected"])
        for pair in pairs
    ]
    margins: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(pairs), batch_size):
        chosen_logps = sequence_logprobs(
            model, chosen[start : start + batch_size], tokenizer.pad_id, device
        )
        rejected_logps = sequence_logprobs(
            model, rejected[start : start + batch_size], tokenizer.pad_id, device
        )
        margins.append((chosen_logps - rejected_logps).cpu())
    if not margins:
        return 0.0, 0.0
    values = torch.cat(margins)
    return float((values > 0).float().mean()), float(values.mean())


def sampled_reference_kl(sequences: Sequence[TrainingSequence]) -> float:
    values: list[float] = []
    for sequence in sequences:
        for policy, reference in zip(sequence.old_logprobs, sequence.reference_logprobs):
            log_ratio = reference - policy
            values.append(math.exp(log_ratio) - log_ratio - 1.0)
    return _mean(values)


def evaluate_model(
    model: SmolLM,
    tokenizer: MicroTokenizer,
    tasks: Sequence[Task],
    *,
    stage: str,
    device: str | torch.device,
    config: EvalConfig | None = None,
    reference: SmolLM | None = None,
    raw_output: str | Path | None = None,
) -> dict[str, object]:
    config = config or EvalConfig()
    model.eval()
    direct_samples: list[TrainingSequence] = []
    episode_samples: list[TrainingSequence] = []
    records: list[dict[str, object]] = []

    for task in tasks:
        direct = sample_group(
            model,
            tokenizer,
            task,
            group_size=1,
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,
            device=device,
        )[0]
        episode = sample_episode(
            model,
            tokenizer,
            task,
            max_turns=config.max_turns,
            max_action_tokens=config.max_new_tokens,
            temperature=0.0,
            device=device,
        )
        direct_samples.append(direct)
        episode_samples.append(episode)
        records.append(
            {
                **task.to_dict(),
                "direct_response": direct.response,
                "direct_reward": direct.reward,
                "direct_correct": bool(direct.breakdown and direct.breakdown.correct),
                "direct_valid": bool(direct.breakdown and direct.breakdown.valid_format),
                "episode_response": episode.response,
                "episode_reward": episode.reward,
                "episode_correct": bool(episode.breakdown and episode.breakdown.correct),
                "episode_valid": bool(episode.breakdown and episode.breakdown.valid_format),
                "episode_used_tool": bool(episode.breakdown and episode.breakdown.used_tool),
                "episode_steps": episode.breakdown.steps if episode.breakdown else 0,
            }
        )

    if reference is not None:
        reference.eval()
        attach_reference_logprobs(reference, direct_samples, tokenizer.pad_id, device)

    direct_correct = [bool(sample.breakdown and sample.breakdown.correct) for sample in direct_samples]
    direct_valid = [bool(sample.breakdown and sample.breakdown.valid_format) for sample in direct_samples]
    episode_correct = [bool(sample.breakdown and sample.breakdown.correct) for sample in episode_samples]
    episode_valid = [bool(sample.breakdown and sample.breakdown.valid_format) for sample in episode_samples]
    episode_tools = [bool(sample.breakdown and sample.breakdown.used_tool) for sample in episode_samples]
    episode_tool_attempts = ["<tool>" in sample.response for sample in episode_samples]
    low, high = bootstrap_interval(
        episode_correct, samples=config.bootstrap_samples, seed=config.seed
    )
    preference_accuracy, preference_margin = heldout_preference_metrics(
        model,
        tokenizer,
        tasks,
        device,
        config.preference_batch_size,
        config.seed,
    )

    by_difficulty: dict[str, dict[str, float]] = {}
    for difficulty in ("easy", "medium", "hard"):
        selected = [index for index, task in enumerate(tasks) if task.difficulty == difficulty]
        by_difficulty[difficulty] = {
            "direct_accuracy": _rate([direct_correct[index] for index in selected]),
            "tool_accuracy": _rate([episode_correct[index] for index in selected]),
            "tool_use_rate": _rate([episode_tools[index] for index in selected]),
        }

    metrics: dict[str, object] = {
        "kind": "evaluation",
        "stage": stage,
        "examples": len(tasks),
        "direct_accuracy": _rate(direct_correct),
        "tool_accuracy": _rate(episode_correct),
        "valid_format_rate": _rate(episode_valid),
        "direct_valid_format_rate": _rate(direct_valid),
        "valid_tool_call_rate": _rate(
            [
                episode_tools[index]
                for index in range(len(episode_samples))
                if episode_tool_attempts[index]
            ]
        ),
        "unnecessary_tool_rate": _rate(
            [episode_tools[index] for index, task in enumerate(tasks) if task.difficulty == "easy"]
        ),
        "mean_reward": _mean([sample.reward for sample in episode_samples]),
        "mean_steps": _mean(
            [sample.breakdown.steps if sample.breakdown else 0 for sample in episode_samples]
        ),
        "mean_generated_tokens": _mean(
            [sum(sample.policy_mask) for sample in episode_samples]
        ),
        "preference_accuracy": preference_accuracy,
        "preference_margin": preference_margin,
        "sampled_reference_kl": sampled_reference_kl(direct_samples) if reference else 0.0,
        "tool_accuracy_ci95": [low, high],
        "by_difficulty": by_difficulty,
    }

    if raw_output is not None:
        output = Path(raw_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return metrics


def append_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
