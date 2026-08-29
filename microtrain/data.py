"""Procedural data views for SFT, DPO, RLVR, agent RL, and evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .environment import SafeCalculator, Task, direct_answer, tool_call, tool_trajectory


TRAIN_TEMPLATES: dict[str, tuple[str, Callable[[random.Random], tuple[int, ...]]]] = {
    "add": ("{0} + {1}", lambda r: (r.randint(0, 99), r.randint(0, 99))),
    "sub": ("{0} - {1}", lambda r: (r.randint(0, 99), r.randint(0, 99))),
    "mul": ("{0} * {1}", lambda r: (r.randint(2, 20), r.randint(2, 20))),
    "mul_add": ("({0} * {1}) + {2}", lambda r: (r.randint(2, 30), r.randint(2, 30), r.randint(0, 99))),
    "add_mul": ("({0} + {1}) * {2}", lambda r: (r.randint(0, 50), r.randint(0, 50), r.randint(2, 15))),
    "two_products": (
        "({0} * {1}) + ({2} * {3})",
        lambda r: tuple(r.randint(10, 99) for _ in range(4)),
    ),
}

EVAL_TEMPLATES: dict[str, tuple[str, Callable[[random.Random], tuple[int, ...]]]] = {
    "heldout_easy": ("({0} + {1})", lambda r: (r.randint(0, 50), r.randint(0, 50))),
    "heldout_add": ("{0} + ({1} + {2})", lambda r: tuple(r.randint(20, 120) for _ in range(3))),
    "heldout_mixed": (
        "({0} - {1}) * ({2} + {3})",
        lambda r: (r.randint(30, 99), r.randint(0, 29), r.randint(10, 50), r.randint(10, 50)),
    ),
    "heldout_products": (
        "({0} * {1}) - ({2} * {3})",
        lambda r: tuple(r.randint(20, 120) for _ in range(4)),
    ),
}


def difficulty_for(template: str) -> str:
    if template in {"add", "sub", "mul", "heldout_easy"}:
        return "easy"
    if template in {"mul_add", "add_mul", "heldout_add"}:
        return "medium"
    return "hard"


def generate_tasks(count: int, seed: int, *, held_out: bool = False) -> list[Task]:
    rng = random.Random(seed)
    templates = EVAL_TEMPLATES if held_out else TRAIN_TEMPLATES
    calculator = SafeCalculator()
    tasks: list[Task] = []
    seen: set[str] = set()
    names = list(templates)
    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not generate enough distinct tasks")
        # Advance the template even when a sampled expression is a duplicate.
        # Otherwise exhausting a small template's finite value space can pin
        # generation to that template forever.
        name = names[(attempts - 1) % len(names)]
        pattern, sample = templates[name]
        expression = pattern.format(*sample(rng))
        if expression in seen:
            continue
        seen.add(expression)
        answer = calculator.evaluate(expression)
        tasks.append(Task(f"{'eval' if held_out else 'train'}-{len(tasks):05d}", expression, answer, difficulty_for(name), name))
    rng.shuffle(tasks)
    return tasks


def wrong_answer(task: Task, rng: random.Random) -> int:
    offsets = [-10, -2, -1, 1, 2, 10]
    return task.answer + rng.choice(offsets)


def prefers_tool(task: Task) -> bool:
    if task.difficulty == "hard":
        return True
    if task.difficulty != "medium":
        return False
    numeric_id = int(task.id.rsplit("-", 1)[-1])
    return numeric_id % 2 == 0


def sft_example(task: Task) -> dict[str, str]:
    completion = tool_trajectory(task) if prefers_tool(task) else direct_answer(task.answer)
    return {"id": task.id, "prompt": task.prompt, "completion": completion}


def preference_example(task: Task, rng: random.Random, index: int) -> dict[str, str]:
    category = ("correctness", "format", "strategy")[index % 3]
    tool_preferred = prefers_tool(task)
    preferred = tool_trajectory(task) if tool_preferred else direct_answer(task.answer)
    if category == "correctness":
        chosen = preferred
        if tool_preferred:
            rejected = (
                tool_call(task.expression)
                + f"\nTool: <result>{task.answer}</result>\nAssistant:"
                + direct_answer(wrong_answer(task, rng))
            )
        else:
            rejected = direct_answer(wrong_answer(task, rng))
    elif category == "format":
        chosen = preferred
        rejected = preferred.replace("<answer>", "<answr>").replace("</answer>", "</answr>")
        if tool_preferred:
            rejected = rejected.replace("<tool>", "<tol>").replace("</tool>", "</tol>")
    else:
        direct = direct_answer(task.answer)
        with_tool = tool_trajectory(task)
        chosen, rejected = (with_tool, direct) if tool_preferred else (direct, with_tool)
    return {
        "id": task.id,
        "prompt": task.prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
    }


@dataclass(frozen=True)
class ManifestSizes:
    sft: int = 1_024
    dpo: int = 1_024
    rlvr: int = 512
    agent_rl: int = 512
    eval: int = 300


def build_manifest(seed: int = 42, sizes: ManifestSizes | None = None) -> dict[str, object]:
    sizes = sizes or ManifestSizes()
    # Easy tasks form the single-turn arithmetic frontier. Easy and medium
    # tasks together form the tool-choice frontier: the former teach restraint
    # while the latter provide problems where calculator use can pay off.
    train_count = max(sizes.sft, sizes.dpo, sizes.rlvr * 3, sizes.agent_rl * 3)
    train = generate_tasks(train_count, seed, held_out=False)
    evaluation = generate_tasks(sizes.eval, seed + 1, held_out=True)
    preference_rng = random.Random(seed + 2)
    manifest: dict[str, object] = {
        "version": 1,
        "seed": seed,
        "sizes": asdict(sizes),
        "protocol": "xml-calculator-v1",
        "sft": [sft_example(task) for task in train[: sizes.sft]],
        "dpo": [
            preference_example(task, preference_rng, index)
            for index, task in enumerate(train[: sizes.dpo])
        ],
        "rlvr": [
            task.to_dict()
            for task in train
            if task.difficulty == "easy"
        ][: sizes.rlvr],
        "agent_rl": [
            task.to_dict()
            for task in train
            if task.difficulty in {"easy", "medium"}
        ][: sizes.agent_rl],
        "eval": [task.to_dict() for task in evaluation],
    }
    expected_sizes = {
        "sft": sizes.sft,
        "dpo": sizes.dpo,
        "rlvr": sizes.rlvr,
        "agent_rl": sizes.agent_rl,
        "eval": sizes.eval,
    }
    for view, expected in expected_sizes.items():
        actual = len(manifest[view])  # type: ignore[arg-type]
        if actual != expected:
            raise RuntimeError(f"could not fill {view} view: expected {expected}, generated {actual}")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def read_manifest(path: str | Path) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("sha256", None)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if expected != actual:
        raise ValueError(f"manifest hash mismatch: expected {expected}, computed {actual}")
    manifest["sha256"] = expected
    return manifest
