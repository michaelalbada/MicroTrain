from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtrain.checkpoint import MODEL_PRESETS
from microtrain.run import (
    aggregate_training_metrics,
    create_data,
    default_run_dir,
    grpo_stage_config,
    highest_available_stage,
    prerequisite_actions,
    tasks_from,
    validate_base_selection,
    validate_manifest_mode,
)


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


def _checkpoint(run_dir: Path, stage: str) -> None:
    destination = run_dir / "checkpoints" / stage
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model.safetensors").write_bytes(b"checkpoint")


def test_prerequisites_are_planned_in_curriculum_order(tmp_path: Path) -> None:
    assert prerequisite_actions(tmp_path, "dpo", include_target=False) == [
        "base",
        "data",
        "sft",
    ]
    assert prerequisite_actions(tmp_path, "agent_rl", include_target=True) == [
        "base",
        "data",
        "sft",
        "dpo",
        "rlvr",
        "agent_rl",
    ]


def test_missing_parent_rebuilds_downstream_prerequisites(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "base")
    (tmp_path / "data_manifest.json").write_text("{}")
    _checkpoint(tmp_path, "sft")
    _checkpoint(tmp_path, "rlvr")
    assert prerequisite_actions(tmp_path, "agent_rl", include_target=False) == [
        "dpo",
        "rlvr",
    ]
    assert highest_available_stage(tmp_path) == "rlvr"


def test_model_presets_use_separate_default_run_directories() -> None:
    assert default_run_dir("135m") == Path("runs/default")
    assert default_run_dir("360m") == Path("runs/360m")


def test_run_directory_rejects_a_different_base_model(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "base")
    metadata = {
        "source": MODEL_PRESETS["135m"].model_id,
        "revision": MODEL_PRESETS["135m"].revision,
    }
    (tmp_path / "checkpoints" / "base" / "microtrain.json").write_text(
        json.dumps(metadata)
    )
    with pytest.raises(RuntimeError, match="different --run-dir"):
        validate_base_selection(
            tmp_path,
            MODEL_PRESETS["360m"],
            MODEL_PRESETS["360m"].revision,
        )


def test_agent_rl_uses_stable_balanced_rollout_settings() -> None:
    config = grpo_stage_config("agent_rl", smoke=False, seed=42)
    assert config.prompts_per_step == 3
    assert config.temperature == 0.9
    assert config.kl_beta >= 0.1
    assert config.tool_guard_steps == 5


def test_training_summary_aggregates_all_updates() -> None:
    summary = aggregate_training_metrics(
        [
            {"step": 1, "loss": 2.0, "correct_rate": 0.25},
            {"step": 2, "loss": 1.0, "correct_rate": 0.75},
        ]
    )
    assert summary == {
        "steps": 2,
        "mean_loss": 1.5,
        "mean_correct_rate": 0.5,
    }
