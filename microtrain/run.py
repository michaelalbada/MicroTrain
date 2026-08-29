"""Command-line orchestration for the cumulative microTrain curriculum."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import torch

from .checkpoint import (
    BASE_MODEL_ID,
    BASE_REVISION,
    checkpoint_metadata,
    download_base,
    load_model,
    load_tokenizer,
    save_checkpoint,
)
from .data import ManifestSizes, build_manifest, read_manifest, write_manifest
from .dpo import DPOConfig, train_dpo
from .environment import SafeCalculator, Task
from .eval import EvalConfig, append_metrics, evaluate_model
from .grpo import GRPOConfig, train_grpo
from .plot import write_capability_svg
from .sft import SFTConfig, train_sft


STAGES = ("base", "sft", "dpo", "rlvr", "agent_rl")
PARENTS = {"sft": "base", "dpo": "sft", "rlvr": "dpo", "agent_rl": "rlvr"}


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def checkpoint_dir(run_dir: Path, stage: str) -> Path:
    return run_dir / "checkpoints" / stage


def require_checkpoint(run_dir: Path, stage: str) -> Path:
    checkpoint = checkpoint_dir(run_dir, stage)
    if not (checkpoint / "model.safetensors").exists() and not list(checkpoint.glob("*.safetensors")):
        raise FileNotFoundError(f"missing {stage} checkpoint; run that stage first")
    return checkpoint


def checkpoint_exists(run_dir: Path, stage: str) -> bool:
    checkpoint = checkpoint_dir(run_dir, stage)
    return (checkpoint / "model.safetensors").is_file() or bool(list(checkpoint.glob("*.safetensors")))


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "data_manifest.json"


def tasks_from(manifest: dict[str, object], key: str) -> list[Task]:
    values = manifest[key]
    assert isinstance(values, list)
    tasks = [Task.from_dict(value) for value in values]  # type: ignore[arg-type]
    calculator = SafeCalculator()
    for task in tasks:
        recomputed = calculator.evaluate(task.expression)
        if recomputed != task.answer:
            raise ValueError(
                f"manifest answer mismatch for {task.id}: stored {task.answer}, computed {recomputed}"
            )
    return tasks


def prepare(run_dir: Path, device: torch.device, dtype: torch.dtype, revision: str) -> None:
    destination = checkpoint_dir(run_dir, "base")
    download_base(destination, revision=revision)
    model = load_model(destination, device=device, dtype=dtype).eval()
    tokenizer = load_tokenizer(destination)
    token_ids = tokenizer.encode("Arithmetic is", bos=True)[:8]
    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)
    with torch.inference_mode():
        full = model(tokens)
        cache = None
        cached_parts = []
        for position in range(tokens.shape[1]):
            logits, cache = model.forward_cached(tokens[:, position : position + 1], cache)
            cached_parts.append(logits)
        cached = torch.cat(cached_parts, dim=1)
    # Full-sequence and token-at-a-time SDPA use different matrix shapes, so
    # their floating-point accumulation order is not bit-identical.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-3
    max_difference = float((full.float() - cached.float()).abs().max())
    if max_difference > tolerance:
        raise RuntimeError(f"cached decoding parity failed: max difference {max_difference}")
    metadata = {
        "stage": "base",
        "source": BASE_MODEL_ID,
        "revision": revision,
        "cached_forward_max_difference": max_difference,
    }
    (destination / "microtrain.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"prepared {BASE_MODEL_ID} at {destination} (cache diff {max_difference:.3g})")


def data_sizes(smoke: bool) -> ManifestSizes:
    return ManifestSizes(sft=12, dpo=12, rlvr=8, agent_rl=8, eval=6) if smoke else ManifestSizes()


def create_data(run_dir: Path, seed: int, smoke: bool) -> dict[str, object]:
    sizes = data_sizes(smoke)
    manifest = build_manifest(seed, sizes)
    destination = manifest_path(run_dir)
    if destination.exists():
        existing = read_manifest(destination)
        if existing["sha256"] != manifest["sha256"]:
            raise FileExistsError(
                f"{destination} already contains a different manifest; use a new --run-dir"
            )
    write_manifest(destination, manifest)
    print(f"wrote {destination} ({manifest['sha256']})")
    return manifest


def load_run_manifest(run_dir: Path) -> dict[str, object]:
    path = manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError("missing data manifest; run `python -m microtrain.run data` first")
    return read_manifest(path)


def validate_manifest_mode(manifest: dict[str, object], smoke: bool) -> None:
    expected = asdict(data_sizes(smoke))
    if manifest.get("sizes") != expected:
        mode = "smoke" if smoke else "full"
        raise RuntimeError(f"run manifest is not a {mode} manifest; use a matching --run-dir")


def record_training_metrics(run_dir: Path, stage: str, values: list[dict[str, float | int]]) -> None:
    for value in values:
        append_metrics(run_dir / "metrics.jsonl", {"kind": "training", "stage": stage, **value})


def train_stage(
    stage: str,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    smoke: bool,
    steps: int | None,
) -> None:
    manifest = load_run_manifest(run_dir)
    validate_manifest_mode(manifest, smoke)
    seed_torch(seed)
    parent_stage = PARENTS[stage]
    parent_path = require_checkpoint(run_dir, parent_stage)
    parent_metadata = checkpoint_metadata(parent_path)
    parent_data_hash = parent_metadata.get("data_sha256")
    if parent_data_hash is not None and parent_data_hash != manifest["sha256"]:
        raise RuntimeError(
            f"{parent_stage} checkpoint was trained with a different manifest; use a new run directory"
        )
    model = load_model(parent_path, device=device, dtype=dtype)
    tokenizer = load_tokenizer(parent_path)

    training_config: object
    if stage == "sft":
        config = SFTConfig(steps=1, batch_size=2, log_every=1, seed=seed) if smoke else SFTConfig(seed=seed)
        if steps is not None:
            config = replace(config, steps=steps)
        values = manifest["sft"]
        assert isinstance(values, list)
        metrics = train_sft(model, tokenizer, values, config, device)  # type: ignore[arg-type]
        training_config = config
    elif stage == "dpo":
        reference = load_model(parent_path, device=device, dtype=dtype)
        config = DPOConfig(steps=1, batch_size=2, log_every=1, seed=seed) if smoke else DPOConfig(seed=seed)
        if steps is not None:
            config = replace(config, steps=steps)
        values = manifest["dpo"]
        assert isinstance(values, list)
        metrics = train_dpo(model, reference, tokenizer, values, config, device)  # type: ignore[arg-type]
        training_config = config
    elif stage in {"rlvr", "agent_rl"}:
        reference = load_model(parent_path, device=device, dtype=dtype)
        config = (
            GRPOConfig(
                steps=1,
                prompts_per_step=1,
                group_size=2,
                update_epochs=1,
                max_new_tokens=8,
                max_turns=2,
                log_every=1,
                seed=seed,
            )
            if smoke
            else GRPOConfig(seed=seed)
        )
        if stage == "agent_rl":
            # Tool decisions are lower entropy than answer digits after SFT.
            # Explore them more aggressively while retaining a substantial KL
            # constraint against the RLVR checkpoint.
            config = replace(config, temperature=1.5, kl_beta=0.05)
        if steps is not None:
            config = replace(config, steps=steps)
        tasks = tasks_from(manifest, stage)
        metrics = train_grpo(
            model,
            reference,
            tokenizer,
            tasks,
            config,
            device,
            tools=stage == "agent_rl",
        )
        training_config = config
    else:
        raise ValueError(f"unknown training stage: {stage}")

    destination = checkpoint_dir(run_dir, stage)
    save_checkpoint(
        destination,
        model,
        tokenizer_source=parent_path,
        config_source=parent_path,
        metadata={
            "stage": stage,
            "parent": parent_stage,
            "parent_metadata": parent_metadata,
            "data_sha256": manifest["sha256"],
            "seed": seed,
            "training_config": asdict(training_config),  # type: ignore[arg-type]
            "final_training_metrics": metrics[-1] if metrics else {},
        },
    )
    record_training_metrics(run_dir, stage, metrics)
    print(f"trained {stage}; wrote {destination}")


def evaluate_stage(
    stage: str,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    smoke: bool,
    eval_limit: int | None = None,
) -> dict[str, object]:
    manifest = load_run_manifest(run_dir)
    validate_manifest_mode(manifest, smoke)
    path = require_checkpoint(run_dir, stage)
    metadata = checkpoint_metadata(path)
    data_hash = metadata.get("data_sha256")
    if data_hash is not None and data_hash != manifest["sha256"]:
        raise RuntimeError(
            f"{stage} checkpoint was trained with a different manifest; use its original run directory"
        )
    model = load_model(path, device=device, dtype=dtype)
    tokenizer = load_tokenizer(path)
    reference = None
    if stage in PARENTS:
        reference = load_model(require_checkpoint(run_dir, PARENTS[stage]), device=device, dtype=dtype)
    tasks = tasks_from(manifest, "eval")
    if eval_limit is not None:
        tasks = tasks[:eval_limit]
    config = EvalConfig(max_new_tokens=8, bootstrap_samples=50) if smoke else EvalConfig()
    metrics = evaluate_model(
        model,
        tokenizer,
        tasks,
        stage=stage,
        device=device,
        config=config,
        reference=reference,
        raw_output=run_dir / f"eval_{stage}.jsonl",
    )
    append_metrics(run_dir / "metrics.jsonl", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def evaluate_available(
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    smoke: bool,
    eval_limit: int | None = None,
) -> list[dict[str, object]]:
    results = [
        evaluate_stage(stage, run_dir, device, dtype, smoke, eval_limit)
        for stage in STAGES
        if checkpoint_exists(run_dir, stage)
    ]
    if results:
        write_capability_svg(run_dir / "metrics.jsonl", run_dir / "capability_curve.svg")
    return results


def write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    values = {
        "base_model": BASE_MODEL_ID,
        "base_revision": args.revision,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "smoke": args.smoke,
        "eval_limit": args.eval_limit,
    }
    (run_dir / "config.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def run_all(args: argparse.Namespace, run_dir: Path, device: torch.device, dtype: torch.dtype) -> None:
    if not checkpoint_exists(run_dir, "base"):
        prepare(run_dir, device, dtype, args.revision)
    create_data(run_dir, args.seed, args.smoke)
    evaluate_stage("base", run_dir, device, dtype, args.smoke, args.eval_limit)
    for stage in ("sft", "dpo", "rlvr", "agent_rl"):
        train_stage(stage, run_dir, device, dtype, args.seed, args.smoke, args.steps)
        evaluate_stage(stage, run_dir, device, dtype, args.smoke, args.eval_limit)
    write_capability_svg(run_dir / "metrics.jsonl", run_dir / "capability_curve.svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "data", "sft", "dpo", "rlvr", "agent-rl", "eval", "plot", "all"],
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--steps", type=int, help="override the selected stage's update count")
    parser.add_argument("--stage", choices=STAGES, help="stage to evaluate")
    parser.add_argument("--eval-limit", type=int, help="evaluate a deterministic prefix of the manifest")
    parser.add_argument("--smoke", action="store_true", help="use tiny data and one training update")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_torch(args.seed)
    write_run_config(run_dir, args, device, dtype)
    print(f"device={device} dtype={dtype}")

    commands: dict[str, Callable[[], object]] = {
        "prepare": lambda: prepare(run_dir, device, dtype, args.revision),
        "data": lambda: create_data(run_dir, args.seed, args.smoke),
        "sft": lambda: train_stage("sft", run_dir, device, dtype, args.seed, args.smoke, args.steps),
        "dpo": lambda: train_stage("dpo", run_dir, device, dtype, args.seed, args.smoke, args.steps),
        "rlvr": lambda: train_stage("rlvr", run_dir, device, dtype, args.seed, args.smoke, args.steps),
        "agent-rl": lambda: train_stage("agent_rl", run_dir, device, dtype, args.seed, args.smoke, args.steps),
        "eval": lambda: (
            evaluate_stage(args.stage, run_dir, device, dtype, args.smoke, args.eval_limit)
            if args.stage
            else evaluate_available(run_dir, device, dtype, args.smoke, args.eval_limit)
        ),
        "plot": lambda: write_capability_svg(run_dir / "metrics.jsonl", run_dir / "capability_curve.svg"),
        "all": lambda: run_all(args, run_dir, device, dtype),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
