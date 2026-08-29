from __future__ import annotations

import json
from pathlib import Path

from microtrain.plot import write_capability_svg


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

