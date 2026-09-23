"""Token accounting and readable single-turn/multi-turn rollout loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .environment import ArithmeticEnv, Task
from .model import SmolLM
from .rewards import RewardBreakdown, episode_reward, single_turn_reward
from .tokenizer import MicroTokenizer


@dataclass
class TrainingSequence:
    prompt: str
    token_ids: list[int]
    policy_mask: list[bool]
    old_logprobs: list[float] = field(default_factory=list)
    reference_logprobs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    reward: float = 0.0
    response: str = ""
    breakdown: RewardBreakdown | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.policy_mask):
            raise ValueError("token_ids and policy_mask must have equal length")
        if self.policy_mask and self.policy_mask[0]:
            raise ValueError("the first token cannot be policy-generated because it has no predecessor")
        generated = sum(self.policy_mask)
        if self.old_logprobs and len(self.old_logprobs) != generated:
            raise ValueError("old_logprobs must have one value per generated token")
        if self.reference_logprobs and len(self.reference_logprobs) != generated:
            raise ValueError("reference_logprobs must have one value per generated token")
        if self.entropies and len(self.entropies) != generated:
            raise ValueError("entropies must have one value per generated token")


def prompt_response_sequence(
    tokenizer: MicroTokenizer,
    prompt: str,
    response: str,
    *,
    append_eos: bool = True,
) -> TrainingSequence:
    prompt_ids = tokenizer.encode(prompt, bos=True)
    response_ids = tokenizer.encode(response, eos=append_eos)
    return TrainingSequence(
        prompt=prompt,
        token_ids=prompt_ids + response_ids,
        policy_mask=[False] * len(prompt_ids) + [True] * len(response_ids),
        response=response,
    )


def conversation_sequence(
    tokenizer: MicroTokenizer,
    prompt: str,
    completion: str,
) -> TrainingSequence:
    """Tokenize a demonstration while masking environment observations.

    Direct responses have no observation and reduce to the ordinary
    completion-only sequence. Tool trajectories are deliberately segmented in
    the same places as live episodes, so only assistant actions receive loss.
    """

    tool_marker = "\nTool:"
    assistant_marker = "\nAssistant:"
    if tool_marker not in completion:
        return prompt_response_sequence(tokenizer, prompt, completion)

    observation_start = completion.index(tool_marker)
    observation_end = completion.index(assistant_marker, observation_start) + len(
        assistant_marker
    )
    first_action = tokenizer.encode(completion[:observation_start])
    observation = tokenizer.encode(completion[observation_start:observation_end])
    final_action = tokenizer.encode(completion[observation_end:], eos=True)
    prompt_ids = tokenizer.encode(prompt, bos=True)
    return TrainingSequence(
        prompt=prompt,
        token_ids=prompt_ids + first_action + observation + final_action,
        policy_mask=(
            [False] * len(prompt_ids)
            + [True] * len(first_action)
            + [False] * len(observation)
            + [True] * len(final_action)
        ),
        response=completion,
    )


def collate_sequences(
    sequences: Sequence[TrainingSequence],
    pad_id: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if not sequences:
        raise ValueError("cannot collate an empty sequence list")
    length = max(len(sequence.token_ids) for sequence in sequences) - 1
    inputs = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    targets = torch.full_like(inputs, pad_id)
    masks = torch.zeros_like(inputs, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        size = len(sequence.token_ids) - 1
        inputs[row, :size] = torch.tensor(sequence.token_ids[:-1], device=device)
        targets[row, :size] = torch.tensor(sequence.token_ids[1:], device=device)
        masks[row, :size] = torch.tensor(sequence.policy_mask[1:], device=device)
    return inputs, targets, masks


def token_logprobs(
    model: SmolLM,
    input_ids: Tensor,
    targets: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    logits = model(input_ids)
    logits = logits.float() / temperature
    return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def sequence_logprobs(
    model: SmolLM,
    sequences: Sequence[TrainingSequence],
    pad_id: int,
    device: str | torch.device,
) -> Tensor:
    inputs, targets, mask = collate_sequences(sequences, pad_id, device)
    logprobs = token_logprobs(model, inputs, targets)
    return (logprobs * mask).sum(-1)


@torch.inference_mode()
def attach_reference_logprobs(
    reference: SmolLM,
    sequences: Sequence[TrainingSequence],
    pad_id: int,
    device: str | torch.device,
    temperature: float = 1.0,
) -> None:
    inputs, targets, masks = collate_sequences(sequences, pad_id, device)
    logprobs = token_logprobs(reference, inputs, targets, temperature)
    for row, sequence in enumerate(sequences):
        sequence.reference_logprobs = logprobs[row][masks[row]].cpu().tolist()


def _sample_token(logits: Tensor, temperature: float) -> tuple[Tensor, Tensor, Tensor]:
    scaled_logits = logits.float() if temperature <= 0 else logits.float() / temperature
    logprobs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -(logprobs.exp() * logprobs).sum(-1)
    if temperature <= 0:
        token = logits.argmax(-1)
    else:
        probabilities = logprobs.exp()
        token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    return token, logprobs.gather(-1, token.unsqueeze(-1)).squeeze(-1), entropy


@torch.inference_mode()
def sample_group(
    model: SmolLM,
    tokenizer: MicroTokenizer,
    task: Task,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str | torch.device,
) -> list[TrainingSequence]:
    prompt_ids = tokenizer.encode(task.prompt, bos=True)
    prompt_tensor = torch.tensor(prompt_ids, device=device).unsqueeze(0).repeat(group_size, 1)
    logits, cache = model.forward_cached(prompt_tensor)
    next_logits = logits[:, -1]
    responses: list[list[int]] = [[] for _ in range(group_size)]
    old_logprobs: list[list[float]] = [[] for _ in range(group_size)]
    entropies: list[list[float]] = [[] for _ in range(group_size)]
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        token, token_logp, token_entropy = _sample_token(next_logits, temperature)
        token = torch.where(finished, torch.full_like(token, tokenizer.eos_id), token)
        for row in range(group_size):
            if finished[row]:
                continue
            value = int(token[row])
            responses[row].append(value)
            old_logprobs[row].append(float(token_logp[row]))
            entropies[row].append(float(token_entropy[row]))
            text = tokenizer.decode(responses[row])
            if value == tokenizer.eos_id or "</answer>" in text or "</tool>" in text:
                finished[row] = True
        if bool(finished.all()):
            break
        logits, cache = model.forward_cached(token.unsqueeze(1), cache)
        next_logits = logits[:, -1]

    samples: list[TrainingSequence] = []
    for response_ids, sampled_logprobs, sampled_entropies in zip(responses, old_logprobs, entropies):
        response = tokenizer.decode(response_ids)
        breakdown = single_turn_reward(task, response)
        samples.append(
            TrainingSequence(
                prompt=task.prompt,
                token_ids=prompt_ids + response_ids,
                policy_mask=[False] * len(prompt_ids) + [True] * len(response_ids),
                old_logprobs=sampled_logprobs,
                entropies=sampled_entropies,
                reward=breakdown.total,
                response=response,
                breakdown=breakdown,
            )
        )
    return samples


@torch.inference_mode()
def sample_episode(
    model: SmolLM,
    tokenizer: MicroTokenizer,
    task: Task,
    *,
    max_turns: int,
    max_action_tokens: int,
    temperature: float,
    device: str | torch.device,
) -> TrainingSequence:
    env = ArithmeticEnv(task)
    token_ids = tokenizer.encode(env.prompt, bos=True)
    policy_mask = [False] * len(token_ids)
    old_logprobs: list[float] = []
    entropies: list[float] = []
    transcript = ""
    prompt_tensor = torch.tensor(token_ids, device=device).unsqueeze(0)
    logits, cache = model.forward_cached(prompt_tensor)
    next_logits = logits[:, -1]
    valid = True
    correct = False
    steps = 0

    for _ in range(max_turns):
        steps += 1
        action_ids: list[int] = []
        for _ in range(max_action_tokens):
            token, token_logp, token_entropy = _sample_token(next_logits, temperature)
            value = int(token[0])
            token_ids.append(value)
            policy_mask.append(True)
            action_ids.append(value)
            old_logprobs.append(float(token_logp[0]))
            entropies.append(float(token_entropy[0]))
            logits, cache = model.forward_cached(token.view(1, 1), cache)
            next_logits = logits[:, -1]
            action = tokenizer.decode(action_ids)
            if value == tokenizer.eos_id or "</answer>" in action or "</tool>" in action:
                break

        action = tokenizer.decode(action_ids)
        transcript += action
        result = env.step(action)
        valid = valid and result.valid
        correct = result.correct
        if result.done:
            break
        transcript += result.observation
        observation_ids = tokenizer.encode(result.observation)
        token_ids.extend(observation_ids)
        policy_mask.extend([False] * len(observation_ids))
        observation = torch.tensor(observation_ids, device=device).unsqueeze(0)
        logits, cache = model.forward_cached(observation, cache)
        next_logits = logits[:, -1]
    else:
        valid = False

    breakdown = episode_reward(
        correct=correct,
        valid=valid,
        tool_calls=env.tool_calls,
        steps=steps,
        tool_attempts=env.tool_attempts,
    )
    return TrainingSequence(
        prompt=task.prompt,
        token_ids=token_ids,
        policy_mask=policy_mask,
        old_logprobs=old_logprobs,
        entropies=entropies,
        reward=breakdown.total,
        response=transcript,
        breakdown=breakdown,
    )
