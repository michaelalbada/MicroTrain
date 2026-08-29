"""A deterministic arithmetic environment with a deliberately tiny tool API."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass


ANSWER_PATTERN = re.compile(r"\s*<answer>\s*(-?\d+)\s*</answer>\s*", re.DOTALL)
TOOL_PATTERN = re.compile(r"\s*<tool>\s*(\{.*?\})\s*</tool>\s*", re.DOTALL)


class CalculatorError(ValueError):
    pass


class SafeCalculator:
    def __init__(self, *, max_chars: int = 128, max_nodes: int = 64, max_abs_value: int = 10**9) -> None:
        self.max_chars = max_chars
        self.max_nodes = max_nodes
        self.max_abs_value = max_abs_value

    def evaluate(self, expression: str) -> int:
        if len(expression) > self.max_chars:
            raise CalculatorError("expression is too long")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise CalculatorError("invalid expression") from exc
        if sum(1 for _ in ast.walk(tree)) > self.max_nodes:
            raise CalculatorError("expression is too complex")
        return self._evaluate(tree.body)

    def _bounded(self, value: int) -> int:
        if abs(value) > self.max_abs_value:
            raise CalculatorError("value is outside the allowed range")
        return value

    def _evaluate(self, node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return self._bounded(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self._bounded(-self._evaluate(node.operand))
        if not isinstance(node, ast.BinOp):
            raise CalculatorError("unsupported syntax")
        left = self._evaluate(node.left)
        right = self._evaluate(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if right == 0:
                raise CalculatorError("division by zero")
            if left % right:
                raise CalculatorError("division must have an integer result")
            result = left // right
        else:
            raise CalculatorError("unsupported operator")
        return self._bounded(result)


@dataclass(frozen=True)
class Task:
    id: str
    expression: str
    answer: int
    difficulty: str
    template: str

    @property
    def prompt(self) -> str:
        return f"User: What is {self.expression}?\nAssistant:"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "expression": self.expression,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "template": self.template,
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Task":
        return cls(
            id=str(value["id"]),
            expression=str(value["expression"]),
            answer=int(value["answer"]),
            difficulty=str(value["difficulty"]),
            template=str(value["template"]),
        )


def direct_answer(answer: int) -> str:
    return f" <answer>{answer}</answer>"


def tool_call(expression: str) -> str:
    payload = json.dumps({"expression": expression}, separators=(",", ":"))
    return f" <tool>{payload}</tool>"


def tool_result(result: int) -> str:
    return f"\nTool: <result>{result}</result>\nAssistant:"


def tool_trajectory(task: Task) -> str:
    return tool_call(task.expression) + tool_result(task.answer) + direct_answer(task.answer)


def parse_answer(text: str) -> int | None:
    match = ANSWER_PATTERN.fullmatch(text)
    return int(match.group(1)) if match else None


def parse_tool_call(text: str) -> str | None:
    match = TOOL_PATTERN.fullmatch(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"expression"}:
        return None
    expression = payload["expression"]
    return expression if isinstance(expression, str) else None


@dataclass
class StepResult:
    observation: str
    done: bool
    correct: bool
    valid: bool


class ArithmeticEnv:
    def __init__(self, task: Task, calculator: SafeCalculator | None = None, max_tool_calls: int = 1) -> None:
        self.task = task
        self.calculator = calculator or SafeCalculator()
        self.max_tool_calls = max_tool_calls
        self.tool_calls = 0
        self.done = False

    @property
    def prompt(self) -> str:
        return self.task.prompt

    def step(self, action: str) -> StepResult:
        if self.done:
            raise RuntimeError("episode is already complete")
        answer = parse_answer(action)
        if answer is not None:
            self.done = True
            return StepResult("", True, answer == self.task.answer, True)

        expression = parse_tool_call(action)
        if expression is None or self.tool_calls >= self.max_tool_calls:
            self.done = True
            return StepResult("", True, False, False)
        self.tool_calls += 1
        try:
            result = self.calculator.evaluate(expression)
        except CalculatorError:
            self.done = True
            return StepResult("", True, False, False)
        return StepResult(tool_result(result), False, False, True)

