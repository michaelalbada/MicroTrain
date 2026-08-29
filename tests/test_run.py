from __future__ import annotations

from pathlib import Path

import pytest

from microtrain.run import create_data, tasks_from, validate_manifest_mode


def test_run_manifest_cannot_be_silently_replaced(tmp_path: Path) -> None:
    first = create_data(tmp_path, seed=42, smoke=True)
    assert create_data(tmp_path, seed=42, smoke=True) == first
    with pytest.raises(FileExistsError, match="new --run-dir"):
        create_data(tmp_path, seed=43, smoke=True)
    with pytest.raises(FileExistsError, match="new --run-dir"):
        create_data(tmp_path, seed=42, smoke=False)
    validate_manifest_mode(first, smoke=True)
    with pytest.raises(RuntimeError, match="not a full manifest"):
        validate_manifest_mode(first, smoke=False)


def test_task_answers_are_recomputed_when_loaded() -> None:
    manifest = {
        "eval": [
            {
                "id": "bad",
                "expression": "2 + 3",
                "answer": 6,
                "difficulty": "easy",
                "template": "test",
            }
        ]
    }
    with pytest.raises(ValueError, match="stored 6, computed 5"):
        tasks_from(manifest, "eval")
