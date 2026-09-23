from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from microtrain.checkpoint import (
    BASE_MODEL_ID,
    BASE_REVISION,
    MODEL_PRESETS,
    canonical_state_dict,
    download_base,
    load_model,
    save_checkpoint,
)
from microtrain.model import ModelConfig, SmolLM


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    source = tmp_path / "source"
    source.mkdir()
    raw_config = {**config.to_dict(), "bos_token_id": 0, "eos_token_id": 0}
    (source / "config.json").write_text(json.dumps(raw_config))
    (source / "tokenizer.json").write_text("{}")
    model = SmolLM(config).eval()
    output = tmp_path / "checkpoint"
    save_checkpoint(
        output,
        model,
        tokenizer_source=source,
        config_source=source,
        metadata={"stage": "test"},
    )
    restored = load_model(output).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 5))
    torch.testing.assert_close(model(tokens), restored(tokens))
    assert "lm_head.weight" not in canonical_state_dict(model)
    assert json.loads((output / "microtrain.json").read_text()) == {"stage": "test"}


def test_download_base_fetches_the_pinned_public_checkpoint(tmp_path: Path, monkeypatch) -> None:
    hub = pytest.importorskip("huggingface_hub")
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        destination = Path(kwargs["local_dir"])
        for filename in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (destination / filename).write_text("test")

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    destination = download_base(tmp_path / "base")
    assert destination == tmp_path / "base"
    assert captured["repo_id"] == BASE_MODEL_ID
    assert captured["revision"] == BASE_REVISION


def test_download_base_accepts_the_larger_preset(tmp_path: Path, monkeypatch) -> None:
    hub = pytest.importorskip("huggingface_hub")
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        destination = Path(kwargs["local_dir"])
        for filename in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (destination / filename).write_text("test")

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    preset = MODEL_PRESETS["360m"]
    download_base(
        tmp_path / "base",
        revision=preset.revision,
        model_id=preset.model_id,
    )
    assert captured["repo_id"] == "HuggingFaceTB/SmolLM2-360M"
    assert captured["revision"] == "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"


@pytest.mark.skipif(
    "MICROTRAIN_BASE_CHECKPOINT" not in os.environ,
    reason="set MICROTRAIN_BASE_CHECKPOINT for optional upstream parity validation",
)
def test_optional_transformers_logit_parity() -> None:
    transformers = pytest.importorskip("transformers")
    checkpoint = os.environ["MICROTRAIN_BASE_CHECKPOINT"]
    ours = load_model(checkpoint).eval()
    upstream = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32
    ).eval()
    tokens = torch.tensor([[0, 100, 200, 300]])
    with torch.no_grad():
        expected = upstream(tokens).logits
        actual = ours(tokens)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-4, rtol=2e-4)
