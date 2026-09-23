from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtrain.eval import aggregate_quality_index
from microtrain.plot import read_evaluations, write_capability_svg


def test_quality_index_rewards_balance_and_penalizes_collapse() -> None:
    balanced = aggregate_quality_index(
        direct_accuracy=0.7,
        tool_accuracy=0.8,
        valid_format_rate=0.9,
        preference_accuracy=0.85,
        unnecessary_tool_rate=0.1,
    )
    collapsed = aggregate_quality_index(
        direct_accuracy=0.7,
        tool_accuracy=0.1,
        valid_format_rate=0.9,
        preference_accuracy=0.99,
        unnecessary_tool_rate=0.1,
    )
    expected = 0.76**0.55 * 0.9**0.20 * 0.85**0.15 * 0.9**0.10
    assert balanced == pytest.approx(expected)
    assert 0.0 < collapsed < balanced < 1.0
    assert aggregate_quality_index(
        direct_accuracy=0.0,
        tool_accuracy=0.0,
        valid_format_rate=1.0,
        preference_accuracy=1.0,
        unnecessary_tool_rate=0.0,
    ) == 0.0


def test_plot_is_generated_from_evaluation_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {"kind": "evaluation", "stage": "base", "direct_accuracy": 0.1, "tool_accuracy": 0.2},
        {"kind": "evaluation", "stage": "sft", "direct_accuracy": 0.5, "tool_accuracy": 0.6},
    ]
    metrics.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    output = write_capability_svg(metrics, tmp_path / "plot.svg")
    text = output.read_text()
    assert "microTrain capability progression" in text
    assert "BASE" in text and "SFT" in text
    assert "Quality index" in text

    loaded = read_evaluations(metrics)
    assert all("quality_index" in record for record in loaded)
    assert all("tool_attempt_rate" not in record for record in loaded)
