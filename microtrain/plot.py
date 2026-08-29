"""Generate a dependency-free SVG capability plot from evaluation records."""

from __future__ import annotations

import html
import json
from pathlib import Path


STAGE_ORDER = ["base", "sft", "dpo", "rlvr", "agent_rl"]
SERIES = {
    "Direct accuracy": ("direct_accuracy", "#2563eb"),
    "Tool accuracy": ("tool_accuracy", "#16a34a"),
    "Valid format": ("valid_format_rate", "#9333ea"),
    "Unnecessary tools": ("unnecessary_tool_rate", "#dc2626"),
}


def read_evaluations(path: str | Path) -> list[dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record.get("kind") == "evaluation" and record.get("stage") in STAGE_ORDER:
            latest[str(record["stage"])] = record
    return [latest[stage] for stage in STAGE_ORDER if stage in latest]


def write_capability_svg(metrics_path: str | Path, output_path: str | Path) -> Path:
    records = read_evaluations(metrics_path)
    if not records:
        raise ValueError("metrics file contains no evaluation records")
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 50, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_step = plot_width / max(1, len(records) - 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}.axis{stroke:#94a3b8;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}.line{fill:none;stroke-width:3}.dot{stroke:white;stroke-width:2}</style>',
        '<text x="90" y="30" font-size="22" font-weight="700">microTrain capability progression</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_height * (1.0 - tick / 100)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12">{tick}%</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}"/>')

    for index, record in enumerate(records):
        x = left + index * x_step
        label = html.escape(str(record["stage"]).replace("_", " ").upper())
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle" font-size="12">{label}</text>')

    for series_index, (label, (key, color)) in enumerate(SERIES.items()):
        points = []
        for index, record in enumerate(records):
            x = left + index * x_step
            value = float(record.get(key, 0.0))
            y = top + plot_height * (1.0 - max(0.0, min(1.0, value)))
            points.append((x, y))
        joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline class="line" stroke="{color}" points="{joined}"/>')
        for x, y in points:
            parts.append(f'<circle class="dot" fill="{color}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        legend_x = left + series_index * 190
        parts.append(f'<line x1="{legend_x}" y1="{height-22}" x2="{legend_x+24}" y2="{height-22}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+32}" y="{height-17}" font-size="12">{html.escape(label)}</text>')

    parts.append("</svg>")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n")
    return destination

