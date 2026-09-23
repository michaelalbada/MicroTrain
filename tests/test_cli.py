from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from microtrain.cli import Reporter


def evaluation(stage: str = "sft") -> dict[str, object]:
    return {
        "stage": stage,
        "quality_index": 0.65,
        "direct_accuracy": 0.25,
        "tool_accuracy": 0.75,
        "tool_attempt_rate": 0.6,
        "tool_use_rate": 0.5,
        "valid_format_rate": 0.9,
        "preference_accuracy": 0.8,
        "unnecessary_tool_rate": 0.1,
        "by_difficulty": {
            "easy": {"tool_accuracy": 0.5},
            "medium": {"tool_accuracy": 0.75},
            "hard": {"tool_accuracy": 1.0},
        },
    }


def test_reporter_renders_dashboard_and_compact_metrics() -> None:
    output = io.StringIO()
    console = Console(file=output, width=140, color_system=None, force_terminal=False)
    reporter = Reporter(console=console)
    reporter.header(
        command="all",
        model="SmolLM2-360M",
        run_dir=Path("runs/test"),
        device="cpu",
        dtype="float32",
        smoke=True,
    )
    reporter.prerequisites("DPO training", ["base", "data", "sft"])
    with reporter.training("sft", 10) as update:
        update({"step": 10, "loss": 0.123, "grad_norm": 1.0})
    with reporter.training("agent_rl", 2) as update:
        update(
            {
                "step": 2,
                "reward": 0.5,
                "correct_rate": 0.5,
                "tool_attempt_rate": 0.4,
                "valid_tool_call_rate": 0.3,
                "tool_use_rate": 0.3,
                "kl": 0.01,
            }
        )
    reporter.training_summary(
        "agent_rl",
        {
            "steps": 2,
            "mean_reward": 0.4,
            "mean_correct_rate": 0.5,
            "mean_tool_attempt_rate": 0.4,
            "mean_valid_tool_call_rate": 0.3,
            "mean_tool_use_rate": 0.3,
            "mean_zero_variance_groups": 0.2,
            "mean_kl": 0.01,
        },
    )
    reporter.warning("Agent RL stopped", "no valid calculator calls")
    reporter.success("SFT checkpoint", "runs/test/checkpoints/sft")
    reporter.evaluation_result(evaluation())
    reporter.evaluations([evaluation()])
    reporter.finish(Path("runs/test"), elapsed_seconds=65)
    text = output.getvalue()
    assert "microTrain" in text
    assert "Automatic prerequisites" in text
    assert "Base" in text and "Data" in text and "SFT" in text
    assert "65.0%" in text and "25.0%" in text and "75.0%" in text
    assert "Quality" in text
    assert "Run complete" in text
    assert "Aggregate" in text and "tool executions 30.0%" in text
    assert "Agent RL stopped" in text
    assert "SFT checkpoint, SFT evaluation" in text
    assert "1m 5s" in text
    assert '"by_difficulty"' not in text


def test_json_mode_emits_machine_readable_evaluation() -> None:
    output = io.StringIO()
    reporter = Reporter(
        console=Console(file=output, color_system=None, force_terminal=False),
        json_mode=True,
    )
    reporter.header(
        command="eval",
        model="SmolLM2-135M",
        run_dir=Path("runs/test"),
        device="cpu",
        dtype="float32",
        smoke=True,
    )
    reporter.evaluations([evaluation("base")])
    reporter.finish(Path("runs/test"), elapsed_seconds=1)
    text = output.getvalue()
    assert text.startswith("{")
    assert '"stage": "base"' in text
