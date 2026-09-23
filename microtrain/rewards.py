"""Observable rewards and verifiers for arithmetic responses and episodes."""

from __future__ import annotations

from dataclasses import dataclass

from .environment import Task, parse_answer, parse_tool_call


CORRECT_REWARD = 1.0
VALID_FORMAT_BONUS = 0.1
INVALID_FORMAT_PENALTY = 0.25
TOOL_CALL_COST = 0.2
EXTRA_STEP_COST = 0.02


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    correct: bool
    valid_format: bool
    used_tool: bool
    attempted_tool: bool = False
    invalid_action: bool = False
    steps: int = 1


def single_turn_reward(task: Task, response: str) -> RewardBreakdown:
    answer = parse_answer(response)
    valid = answer is not None
    correct = answer == task.answer if valid else False
    used_tool = parse_tool_call(response) is not None
    attempted_tool = "<tool>" in response
    reward = (CORRECT_REWARD if correct else 0.0) + (
        VALID_FORMAT_BONUS if valid else -VALID_FORMAT_BONUS
    )
    return RewardBreakdown(
        reward,
        correct,
        valid,
        used_tool,
        attempted_tool=attempted_tool,
    )


def episode_reward(
    *,
    correct: bool,
    valid: bool,
    tool_calls: int,
    steps: int,
    tool_attempts: int | None = None,
) -> RewardBreakdown:
    invalid = not valid
    reward = (
        (CORRECT_REWARD if correct else 0.0)
        + (VALID_FORMAT_BONUS if valid else -INVALID_FORMAT_PENALTY)
        - TOOL_CALL_COST * tool_calls
        - EXTRA_STEP_COST * max(0, steps - 1)
    )
    attempts = tool_calls if tool_attempts is None else tool_attempts
    return RewardBreakdown(
        reward,
        correct,
        valid,
        tool_calls > 0,
        attempted_tool=attempts > 0,
        invalid_action=invalid,
        steps=steps,
    )
