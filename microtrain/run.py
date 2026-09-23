"""Command-line orchestration for the cumulative microTrain curriculum."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .checkpoint import (
    DEFAULT_MODEL,
    MODEL_PRESETS,
    ModelPreset,
    checkpoint_metadata,
    download_base,
    load_model,
    load_tokenizer,
    save_checkpoint,
)
from .cli import Reporter, STAGE_LABELS
from .data import ManifestSizes, build_manifest, read_manifest, write_manifest
from .dpo import DPOConfig, train_dpo
from .environment import SafeCalculator, Task
from .eval import EvalConfig, append_metrics, evaluate_model
from .grpo import GRPOConfig, NoValidToolCallsError, train_grpo
from .plot import read_evaluations, write_capability_svg
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


def default_run_dir(model: str) -> Path:
    return Path("runs/default" if model == DEFAULT_MODEL else f"runs/{model}")


def validate_base_selection(run_dir: Path, preset: ModelPreset, revision: str) -> None:
    if not checkpoint_exists(run_dir, "base"):
        return
    metadata = checkpoint_metadata(checkpoint_dir(run_dir, "base"))
    source = metadata.get("source")
    saved_revision = metadata.get("revision")
    if source is not None and source != preset.model_id:
        raise RuntimeError(
            f"{run_dir} contains {source}, not {preset.model_id}; use a different --run-dir"
        )
    if saved_revision is not None and saved_revision != revision:
        raise RuntimeError(
            f"{run_dir} contains revision {saved_revision}, not {revision}; "
            "use a different --run-dir"
        )


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


def prepare(
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    preset: ModelPreset,
    revision: str,
) -> dict[str, object]:
    validate_base_selection(run_dir, preset, revision)
    destination = checkpoint_dir(run_dir, "base")
    download_base(destination, revision=revision, model_id=preset.model_id)
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
        "model": preset.name,
        "source": preset.model_id,
        "revision": revision,
        "cached_forward_max_difference": max_difference,
    }
    (destination / "microtrain.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def data_sizes(smoke: bool) -> ManifestSizes:
    return ManifestSizes(sft=12, dpo=12, rlvr=8, agent_rl=8, eval=6) if smoke else ManifestSizes()


def grpo_stage_config(
    stage: str,
    *,
    smoke: bool,
    seed: int,
    steps: int | None = None,
) -> GRPOConfig:
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
        config = replace(
            config,
            prompts_per_step=3,
            temperature=0.9,
            kl_beta=0.1,
        )
    if steps is not None:
        config = replace(config, steps=steps)
    return config


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
        return existing
    write_manifest(destination, manifest)
    return manifest


def load_run_manifest(run_dir: Path) -> dict[str, object]:
    path = manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(
            "missing data manifest; run `uv run microtrain data` first"
        )
    return read_manifest(path)


def validate_manifest_mode(manifest: dict[str, object], smoke: bool) -> None:
    expected = asdict(data_sizes(smoke))
    if manifest.get("sizes") != expected:
        mode = "smoke" if smoke else "full"
        raise RuntimeError(f"run manifest is not a {mode} manifest; use a matching --run-dir")


def record_training_metrics(run_dir: Path, stage: str, values: list[dict[str, float | int]]) -> None:
    for value in values:
        append_metrics(run_dir / "metrics.jsonl", {"kind": "training", "stage": stage, **value})


def aggregate_training_metrics(
    values: list[dict[str, float | int]],
) -> dict[str, float | int]:
    if not values:
        return {"steps": 0}
    summary: dict[str, float | int] = {"steps": int(values[-1]["step"])}
    for name in values[0]:
        if name == "step":
            continue
        summary_name = name if name.startswith("mean_") else f"mean_{name}"
        summary[summary_name] = sum(float(value[name]) for value in values) / len(values)
    return summary


def record_training_summary(
    run_dir: Path,
    stage: str,
    summary: dict[str, float | int],
    *,
    status: str = "completed",
) -> None:
    append_metrics(
        run_dir / "metrics.jsonl",
        {"kind": "training_summary", "stage": stage, "status": status, **summary},
    )


def train_stage(
    stage: str,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    smoke: bool,
    steps: int | None,
    reporter: Reporter | None = None,
) -> Path:
    reporter = reporter or Reporter()
    manifest = load_run_manifest(run_dir)
    validate_manifest_mode(manifest, smoke)
    seed_torch(seed)
    reporter.section(f"{STAGE_LABELS[stage]} training")
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
    metrics_streamed = False
    if stage == "sft":
        config = SFTConfig(steps=1, batch_size=2, log_every=1, seed=seed) if smoke else SFTConfig(seed=seed)
        if steps is not None:
            config = replace(config, steps=steps)
        values = manifest["sft"]
        assert isinstance(values, list)
        with reporter.training(stage, config.steps) as on_log:
            metrics = train_sft(  # type: ignore[arg-type]
                model, tokenizer, values, config, device, on_log=on_log
            )
        training_config = config
    elif stage == "dpo":
        reference = load_model(parent_path, device=device, dtype=dtype)
        config = DPOConfig(steps=1, batch_size=2, log_every=1, seed=seed) if smoke else DPOConfig(seed=seed)
        if steps is not None:
            config = replace(config, steps=steps)
        values = manifest["dpo"]
        assert isinstance(values, list)
        with reporter.training(stage, config.steps) as on_log:
            metrics = train_dpo(  # type: ignore[arg-type]
                model,
                reference,
                tokenizer,
                values,
                config,
                device,
                on_log=on_log,
            )
        training_config = config
    elif stage in {"rlvr", "agent_rl"}:
        reference = load_model(parent_path, device=device, dtype=dtype)
        # Agent RL uses one prompt from each difficulty and lower-temperature
        # sampling to preserve the strict JSON tool protocol.
        config = grpo_stage_config(
            stage,
            smoke=smoke,
            seed=seed,
            steps=steps,
        )
        tasks = tasks_from(manifest, stage)
        trajectory_path = run_dir / f"train_{stage}_trajectories.jsonl"
        with reporter.training(stage, config.steps) as on_log:
            def report_and_record(record: dict[str, float | int]) -> None:
                on_log(record)
                append_metrics(
                    run_dir / "metrics.jsonl",
                    {"kind": "training", "stage": stage, **record},
                )

            try:
                metrics = train_grpo(
                    model,
                    reference,
                    tokenizer,
                    tasks,
                    config,
                    device,
                    tools=stage == "agent_rl",
                    balanced_difficulties=stage == "agent_rl",
                    trajectory_output=trajectory_path,
                    on_log=report_and_record,
                )
            except NoValidToolCallsError as error:
                if error.metrics:
                    stopped_summary = aggregate_training_metrics(error.metrics)
                    record_training_summary(
                        run_dir,
                        stage,
                        stopped_summary,
                        status="stopped_no_valid_tools",
                    )
                    reporter.training_summary(stage, stopped_summary)
                reporter.warning(
                    "Agent RL stopped",
                    f"{error}\nTrajectories: {trajectory_path}",
                )
                raise SystemExit(2) from None
        metrics_streamed = True
        training_config = config
    else:
        raise ValueError(f"unknown training stage: {stage}")

    aggregate_metrics = aggregate_training_metrics(metrics)
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
            "aggregate_training_metrics": aggregate_metrics,
        },
    )
    if not metrics_streamed:
        record_training_metrics(run_dir, stage, metrics)
    record_training_summary(run_dir, stage, aggregate_metrics)
    reporter.training_summary(stage, aggregate_metrics)
    reporter.success(f"{STAGE_LABELS[stage]} checkpoint", str(destination))
    return destination


def evaluate_stage(
    stage: str,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    smoke: bool,
    eval_limit: int | None = None,
    reporter: Reporter | None = None,
) -> dict[str, object]:
    reporter = reporter or Reporter()
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
    with reporter.evaluation(stage, len(tasks)) as on_progress:
        metrics = evaluate_model(
            model,
            tokenizer,
            tasks,
            stage=stage,
            device=device,
            config=config,
            reference=reference,
            raw_output=run_dir / f"eval_{stage}.jsonl",
            on_progress=on_progress,
        )
    append_metrics(run_dir / "metrics.jsonl", metrics)
    reporter.evaluation_result(metrics)
    return metrics


def evaluate_available(
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    smoke: bool,
    eval_limit: int | None = None,
    reporter: Reporter | None = None,
) -> list[dict[str, object]]:
    reporter = reporter or Reporter()
    results = [
        evaluate_stage(stage, run_dir, device, dtype, smoke, eval_limit, reporter)
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
    preset: ModelPreset,
    revision: str,
) -> None:
    values = {
        "command": args.command,
        "model": preset.name,
        "base_model": preset.model_id,
        "base_revision": revision,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "smoke": args.smoke,
        "eval_limit": args.eval_limit,
    }
    (run_dir / "config.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def prepare_with_ui(
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    preset: ModelPreset,
    revision: str,
    reporter: Reporter,
) -> None:
    reporter.section("Base checkpoint")
    metadata = prepare(run_dir, device, dtype, preset, revision)
    difference = float(metadata["cached_forward_max_difference"])
    reporter.success(
        "Base checkpoint ready",
        f"{preset.model_id}@{revision[:8]}  •  cache parity Δ {difference:.3g}  •  "
        f"{checkpoint_dir(run_dir, 'base')}",
    )


def data_with_ui(run_dir: Path, seed: int, smoke: bool, reporter: Reporter) -> None:
    reporter.section("Data manifest")
    manifest = create_data(run_dir, seed, smoke)
    sizes = manifest["sizes"]
    assert isinstance(sizes, dict)
    reporter.success(
        "Data manifest ready",
        f"SFT {sizes['sft']:,}  •  DPO {sizes['dpo']:,}  •  "
        f"RLVR {sizes['rlvr']:,}  •  agent {sizes['agent_rl']:,}  •  "
        f"eval {sizes['eval']:,}  •  {str(manifest['sha256'])[:12]}",
    )


def training_chain(stage: str, *, include_target: bool) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    stop = STAGES.index(stage) + (1 if include_target else 0)
    return list(STAGES[1:stop])


def highest_available_stage(run_dir: Path) -> str:
    return next(
        (stage for stage in reversed(STAGES) if checkpoint_exists(run_dir, stage)),
        "base",
    )


def prerequisite_actions(
    run_dir: Path,
    stage: str,
    *,
    include_target: bool,
) -> list[str]:
    """Return missing artifacts in execution order, propagating rebuilds downstream."""

    actions: list[str] = []
    rebuild = not checkpoint_exists(run_dir, "base")
    if rebuild:
        actions.append("base")
    if rebuild or not manifest_path(run_dir).exists():
        actions.append("data")
        rebuild = True
    for prerequisite in training_chain(stage, include_target=include_target):
        if rebuild or not checkpoint_exists(run_dir, prerequisite):
            actions.append(prerequisite)
            rebuild = True
    return actions


def ensure_base(
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    preset: ModelPreset,
    revision: str,
    reporter: Reporter,
    *,
    target: str,
) -> None:
    if checkpoint_exists(run_dir, "base"):
        validate_base_selection(run_dir, preset, revision)
        return
    reporter.prerequisites(target, ["base"])
    prepare_with_ui(run_dir, device, dtype, preset, revision, reporter)


def ensure_stage(
    stage: str,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    smoke: bool,
    preset: ModelPreset,
    revision: str,
    reporter: Reporter,
    *,
    include_target: bool,
    target: str,
) -> None:
    validate_base_selection(run_dir, preset, revision)
    actions = prerequisite_actions(run_dir, stage, include_target=include_target)
    reporter.prerequisites(target, actions)
    if "base" in actions:
        prepare_with_ui(run_dir, device, dtype, preset, revision, reporter)
    if "data" in actions:
        data_with_ui(run_dir, seed, smoke, reporter)
    else:
        create_data(run_dir, seed, smoke)  # validate the immutable manifest
    for prerequisite in actions:
        if prerequisite in PARENTS:
            train_stage(
                prerequisite,
                run_dir,
                device,
                dtype,
                seed,
                smoke,
                steps=None,
                reporter=reporter,
            )


def run_all(
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    preset: ModelPreset,
    revision: str,
    reporter: Reporter,
) -> None:
    if not checkpoint_exists(run_dir, "base"):
        prepare_with_ui(run_dir, device, dtype, preset, revision, reporter)
    else:
        validate_base_selection(run_dir, preset, revision)
    data_with_ui(run_dir, args.seed, args.smoke, reporter)
    results = [
        evaluate_stage(
            "base",
            run_dir,
            device,
            dtype,
            args.smoke,
            args.eval_limit,
            reporter,
        )
    ]
    for stage in ("sft", "dpo", "rlvr", "agent_rl"):
        train_stage(
            stage,
            run_dir,
            device,
            dtype,
            args.seed,
            args.smoke,
            args.steps,
            reporter,
        )
        results.append(
            evaluate_stage(
                stage,
                run_dir,
                device,
                dtype,
                args.smoke,
                args.eval_limit,
                reporter,
            )
        )
    plot = write_capability_svg(run_dir / "metrics.jsonl", run_dir / "capability_curve.svg")
    reporter.evaluations(results)
    reporter.success("Capability plot", str(plot))
    reporter.artifacts(run_dir, plot=True)


def run_plot(
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    preset: ModelPreset,
    revision: str,
    reporter: Reporter,
) -> None:
    available = [stage for stage in STAGES if checkpoint_exists(run_dir, stage)]
    if not available:
        ensure_stage(
            "base",
            run_dir,
            device,
            dtype,
            args.seed,
            args.smoke,
            preset,
            revision,
            reporter,
            include_target=True,
            target="plot",
        )
        available = ["base"]
    else:
        # Plotting an existing run is read-only with respect to its curriculum.
        # Its manifest may predate the current data generator but is still the
        # correct manifest for those checkpoints and evaluation records.
        validate_base_selection(run_dir, preset, revision)
    metrics_path = run_dir / "metrics.jsonl"
    evaluated = {
        str(record["stage"])
        for record in read_evaluations(metrics_path)
    } if metrics_path.exists() else set()
    missing = [stage for stage in available if stage not in evaluated]
    reporter.prerequisites("plot", [f"eval:{stage}" for stage in missing])
    for stage in missing:
        evaluate_stage(
            stage,
            run_dir,
            device,
            dtype,
            args.smoke,
            args.eval_limit,
            reporter,
        )
    plot = write_capability_svg(metrics_path, run_dir / "capability_curve.svg")
    reporter.evaluations(read_evaluations(metrics_path))
    reporter.success("Capability plot", str(plot))
    reporter.artifacts(run_dir, plot=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microtrain",
        description=__doc__,
        epilog="Missing prerequisites are detected and run automatically.",
    )
    parser.add_argument(
        "command",
        choices=["prepare", "data", "sft", "dpo", "rlvr", "agent-rl", "eval", "plot", "all"],
    )
    parser.add_argument(
        "--model",
        choices=MODEL_PRESETS,
        default=DEFAULT_MODEL,
        help="base model preset (default: %(default)s)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="artifact directory (default: runs/default for 135m, runs/<model> otherwise)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="override the selected preset's pinned revision")
    parser.add_argument("--steps", type=int, help="override the selected stage's update count")
    parser.add_argument("--stage", choices=STAGES, help="stage to evaluate")
    parser.add_argument("--eval-limit", type=int, help="evaluate a deterministic prefix of the manifest")
    parser.add_argument("--smoke", action="store_true", help="use tiny data and one training update")
    parser.add_argument(
        "--json",
        action="store_true",
        help="disable the dashboard and print evaluation results as JSON",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started_at = time.perf_counter()
    preset = MODEL_PRESETS[args.model]
    revision = args.revision or preset.revision
    run_dir: Path = args.run_dir or default_run_dir(args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_torch(args.seed)
    write_run_config(run_dir, args, device, dtype, preset, revision)
    reporter = Reporter(json_mode=args.json)
    reporter.header(
        command=args.command,
        model=preset.display_name,
        run_dir=run_dir,
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
        smoke=args.smoke,
    )

    if args.command == "prepare":
        prepare_with_ui(run_dir, device, dtype, preset, revision, reporter)
    elif args.command == "data":
        ensure_base(
            run_dir,
            device,
            dtype,
            preset,
            revision,
            reporter,
            target="data generation",
        )
        data_with_ui(run_dir, args.seed, args.smoke, reporter)
    elif args.command in {"sft", "dpo", "rlvr", "agent-rl"}:
        stage = args.command.replace("-", "_")
        ensure_stage(
            stage,
            run_dir,
            device,
            dtype,
            args.seed,
            args.smoke,
            preset,
            revision,
            reporter,
            include_target=False,
            target=f"{STAGE_LABELS[stage]} training",
        )
        train_stage(
            stage,
            run_dir,
            device,
            dtype,
            args.seed,
            args.smoke,
            args.steps,
            reporter,
        )
    elif args.command == "eval":
        if args.stage:
            ensure_stage(
                args.stage,
                run_dir,
                device,
                dtype,
                args.seed,
                args.smoke,
                preset,
                revision,
                reporter,
                include_target=True,
                target=f"{STAGE_LABELS[args.stage]} evaluation",
            )
            results = [
                evaluate_stage(
                    args.stage,
                    run_dir,
                    device,
                    dtype,
                    args.smoke,
                    args.eval_limit,
                    reporter,
                )
            ]
        else:
            highest = highest_available_stage(run_dir)
            ensure_stage(
                highest,
                run_dir,
                device,
                dtype,
                args.seed,
                args.smoke,
                preset,
                revision,
                reporter,
                include_target=True,
                target="evaluation",
            )
            results = evaluate_available(
                run_dir,
                device,
                dtype,
                args.smoke,
                args.eval_limit,
                reporter,
            )
        reporter.evaluations(results)
        reporter.artifacts(run_dir, plot=(run_dir / "capability_curve.svg").exists())
    elif args.command == "plot":
        run_plot(args, run_dir, device, dtype, preset, revision, reporter)
    elif args.command == "all":
        run_all(args, run_dir, device, dtype, preset, revision, reporter)

    reporter.finish(run_dir, elapsed_seconds=time.perf_counter() - started_at)


if __name__ == "__main__":
    main()
