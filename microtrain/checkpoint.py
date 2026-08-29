"""Download and serialize SmolLM2 checkpoints without Transformers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .model import ModelConfig, SmolLM
from .tokenizer import MicroTokenizer


BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
BASE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def download_base(destination: str | Path, revision: str = BASE_REVISION) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install 'huggingface_hub' to download the base checkpoint") from exc

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BASE_MODEL_ID,
        revision=revision,
        local_dir=destination,
        allow_patterns=["config.json", "*.safetensors", *TOKENIZER_FILES],
    )
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing = [filename for filename in required if not (destination / filename).is_file()]
    if missing:
        raise RuntimeError(f"incomplete {BASE_MODEL_ID} download: missing {missing}")
    return destination


def load_config(checkpoint: str | Path) -> tuple[ModelConfig, dict[str, Any]]:
    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    return ModelConfig.from_dict(raw), raw


def load_tokenizer(checkpoint: str | Path) -> MicroTokenizer:
    config, raw = load_config(checkpoint)
    del config
    return MicroTokenizer.from_file(
        Path(checkpoint) / "tokenizer.json",
        bos_id=int(raw.get("bos_token_id", 0)),
        eos_id=int(raw.get("eos_token_id", 0)),
    )


def load_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SmolLM:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Install 'safetensors' to load model weights") from exc

    checkpoint = Path(checkpoint)
    config, _ = load_config(checkpoint)
    model = SmolLM(config).to(dtype=dtype)
    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights found in {checkpoint}")
    state: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        state.update(load_file(str(weight_file), device="cpu"))
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not match handwritten model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(device=device, dtype=dtype)
    return model


def canonical_state_dict(model: SmolLM) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    return state


def save_checkpoint(
    destination: str | Path,
    model: SmolLM,
    *,
    tokenizer_source: str | Path,
    config_source: str | Path,
    metadata: dict[str, Any],
) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("Install 'safetensors' to save model weights") from exc

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw_config = json.loads((Path(config_source) / "config.json").read_text())
    raw_config.update(model.config.to_dict())
    (destination / "config.json").write_text(json.dumps(raw_config, indent=2, sort_keys=True) + "\n")
    for filename in TOKENIZER_FILES:
        source = Path(tokenizer_source) / filename
        if source.exists():
            shutil.copyfile(source, destination / filename)
    save_file(canonical_state_dict(model), str(destination / "model.safetensors"))
    (destination / "microtrain.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return destination


def checkpoint_metadata(checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint) / "microtrain.json"
    return json.loads(path.read_text()) if path.exists() else {}
