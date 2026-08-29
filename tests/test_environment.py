from __future__ import annotations

import random

import pytest

from microtrain.data import (
    ManifestSizes,
    build_manifest,
    generate_tasks,
    preference_example,
    read_manifest,
    write_manifest,
)
from microtrain.environment import (
    ArithmeticEnv,
    CalculatorError,
    SafeCalculator,
    Task,
    direct_answer,
    parse_answer,
    parse_tool_call,
    tool_call,
)
from microtrain.rewards import episode_reward


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3", 5), ("(37 * 48) + 19", 1795), ("-8 + 3", -5), ("12 / 3", 4)],
)
def test_safe_calculator(expression: str, expected: int) -> None:
    assert SafeCalculator().evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo nope')",
        "1 / 0",
        "5 / 2",
        "True + 1",
        "2 ** 8",
        "[1, 2, 3]",
    ],
)
def test_safe_calculator_rejects_non_allowlisted_input(expression: str) -> None:
    with pytest.raises(CalculatorError):
        SafeCalculator().evaluate(expression)


def test_protocol_parsers_are_strict() -> None:
    assert parse_answer(" <answer>-12</answer>") == -12
    assert parse_answer("answer: -12") is None
    assert parse_tool_call(tool_call("2 + 3")) == "2 + 3"
    assert parse_tool_call('<tool>{"expression":"2+3","extra":1}</tool>') is None


def test_environment_executes_one_tool_call_then_accepts_answer() -> None:
    task = Task("x", "37 * 48 + 19", 1795, "hard", "test")
    env = ArithmeticEnv(task)
    result = env.step(tool_call(task.expression))
    assert result.valid and not result.done
    result = env.step(direct_answer(task.answer))
    assert result.valid and result.done and result.correct
    assert env.tool_calls == 1


def test_episode_reward_makes_tool_use_a_real_choice() -> None:
    correct_direct = episode_reward(correct=True, valid=True, tool_calls=0, steps=1).total
    correct_tool = episode_reward(correct=True, valid=True, tool_calls=1, steps=2).total
    wrong_direct = episode_reward(correct=False, valid=True, tool_calls=0, steps=1).total
    assert correct_direct > correct_tool > wrong_direct


def test_manifest_is_deterministic_and_structurally_held_out(tmp_path) -> None:
    sizes = ManifestSizes(sft=12, dpo=12, rlvr=8, agent_rl=8, eval=6)
    first = build_manifest(42, sizes)
    second = build_manifest(42, sizes)
    assert first == second
    train_templates = {task.template for task in generate_tasks(30, 42)}
    eval_templates = {task.template for task in generate_tasks(30, 43, held_out=True)}
    assert train_templates.isdisjoint(eval_templates)
    assert {task.difficulty for task in generate_tasks(40, 43, held_out=True)} == {
        "easy",
        "medium",
        "hard",
    }
    assert len(first["rlvr"]) == sizes.rlvr
    assert {example["difficulty"] for example in first["rlvr"]} == {"easy"}
    assert len(first["agent_rl"]) == sizes.agent_rl
    assert {example["difficulty"] for example in first["agent_rl"]} == {"easy", "medium"}
    path = tmp_path / "manifest.json"
    write_manifest(path, first)
    assert read_manifest(path) == first


def test_strategy_preferences_teach_when_not_to_use_the_tool() -> None:
    easy = Task("train-00001", "2 + 3", 5, "easy", "add")
    hard = Task("train-00002", "20 * 30 + 4 * 5", 620, "hard", "two_products")
    easy_pair = preference_example(easy, random.Random(1), index=2)
    hard_pair = preference_example(hard, random.Random(1), index=2)
    assert "<tool>" not in easy_pair["chosen"] and "<tool>" in easy_pair["rejected"]
    assert "<tool>" in hard_pair["chosen"] and "<tool>" not in hard_pair["rejected"]


def test_manifest_hash_detects_mutation(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, build_manifest(42, ManifestSizes(4, 4, 4, 4, 4)))
    text = path.read_text().replace("What is", "Compute", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="hash mismatch"):
        read_manifest(path)


def test_large_manifest_does_not_stall_on_finite_template_spaces() -> None:
    manifest = build_manifest(42, ManifestSizes(8, 8, 512, 8, 8))
    assert len(manifest["rlvr"]) == 512
