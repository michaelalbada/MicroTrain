# microTrain

Modern LLM post-training, stripped to the essentials.

`microTrain` starts with a public **SmolLM2 base** checkpoint and follows one model through a cumulative capability curriculum. SmolLM2-135M is the default; SmolLM2-360M runs the identical teaching code with more capability headroom.

```text
base -> SFT -> DPO -> GRPO/RLVR -> tool-use RL -> evaluation
```

The same procedural arithmetic environment supplies demonstrations, preference pairs, verified rewards, calculator episodes, and held-out evaluation. The goal is to inspect what each stage changes—including regressions and degenerate policies—not to provide another extensible training framework.

## What is deliberately absent

- No Transformers model, generation, or Trainer APIs.
- No TRL, VERL, Axolotl, DeepSpeed, or PEFT.
- No generic trainer or algorithm registry.
- No arbitrary-model compatibility layer.

The runtime boundary is `torch`, `huggingface_hub`, `safetensors`, `tokenizers`, and `rich` for terminal presentation. The released Hugging Face checkpoint supplies weights and tokenizer artifacts; [`microtrain/model.py`](microtrain/model.py) implements the transformer itself.

Both base presets are pinned: `HuggingFaceTB/SmolLM2-135M` at `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` and `HuggingFaceTB/SmolLM2-360M` at `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`. `prepare` downloads the selected revision, loads every tensor into the handwritten model, and checks full-forward versus cached-decoding logits.

## Curriculum

| Stage | What changes |
|---|---|
| Base | Establish the pretrained model's starting behavior. |
| SFT | Learn answer tags, tool syntax, and demonstrations. |
| DPO | Prefer correct, well-formed, and efficient responses. |
| RLVR | Improve exact correctness from an executable verifier. |
| Tool-use RL | Optimize complete calculator episodes with the same GRPO loss. |

Every checkpoint is evaluated on the same structurally held-out manifest. Results include direct and tool-enabled accuracy, format validity, preference accuracy, tool-use cost, and raw generations.

## Install

```bash
uv sync
```

`uv sync` creates the managed `.venv`, installs microTrain, and includes the development group used for tests and upstream parity. Use `uv sync --no-dev` for runtime dependencies only.

The first `prepare` needs Hugging Face network access. Later runs can reuse the local Hub cache. No Hugging Face token is required for this public checkpoint.

## Run

```bash
uv run microtrain prepare
uv run microtrain data
uv run microtrain sft
uv run microtrain dpo
uv run microtrain rlvr
uv run microtrain agent-rl
uv run microtrain eval --stage agent_rl
uv run microtrain plot
```

These commands do not need to be run in order. microTrain inspects the run directory, announces missing prerequisites, and executes only the required parent stages before continuing. For example, starting with `dpo` in an empty run directory automatically runs Base → Data → SFT first. `eval --stage agent_rl` resolves the complete checkpoint chain, while `plot` backfills evaluations only for checkpoints that exist.

Or run the cumulative curriculum:

```bash
uv run microtrain all
```

Run the same curriculum with the larger model:

```bash
uv run microtrain all --model 360m
```

The default 135M run remains in `runs/default`; 360M uses `runs/360m` automatically. An explicit `--run-dir` still takes precedence, and microTrain rejects a directory containing a different base model or revision.

Use `--smoke` for tiny manifests and one optimizer update per stage. Smoke mode proves execution and tensor invariants; it is not expected to produce capability gains.

The default run writes checkpoints, raw generations, metrics, and `capability_curve.svg` under `runs/default/`.

The terminal UI shows live training/evaluation progress, compact stage results, the cumulative capability table, artifact locations, and a final summary of everything completed. Machine-readable output remains available explicitly:

```bash
uv run microtrain eval --stage sft --json
```

Useful development overrides are intentionally few:

```bash
uv run microtrain all --smoke --device cpu
uv run microtrain sft --steps 20
uv run microtrain eval --eval-limit 32   # all available checkpoints
```

`uv run python -m microtrain.run ...` remains supported when a module-form command is preferable.

Each run directory owns one immutable seeded manifest. Use a new `--run-dir` when changing the seed, smoke/full data size, or curriculum version; this prevents accidentally comparing checkpoints trained on different data. Runs created before the balanced Agent-RL curriculum require a new directory, for example `--run-dir runs/360m-balanced`.

## Runtime expectations

Both presets are full-finetuned, and DPO/GRPO keep a frozen reference model alongside the trainable policy. The 360M preset therefore needs materially more memory and time than 135M; it is the better capability experiment, while 135M remains the faster code-reading and smoke-test path. CPU is suitable for tests and smoke runs; a full curriculum is much more practical on CUDA or Apple Silicon. `auto` selects CUDA, then MPS, then CPU, and uses bfloat16 only when CUDA reports native support. Exact runtime depends heavily on the device, so measured artifacts—not hard-coded percentages or timing claims—are the source of truth.

For an optional upstream parity check:

```bash
MICROTRAIN_BASE_CHECKPOINT=runs/default/checkpoints/base uv run pytest tests/test_checkpoint.py -k parity
```

## Checkpoints

Each stage saves:

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors
microtrain.json
```

Parameter names and tensor shapes remain compatible with the SmolLM2 base architecture. `microtrain.json` records the stage, parent checkpoint, data hash, seed, hyperparameters, and final training metrics. This is the interchange contract for the separate `microServe` project.

SFT and DPO mask user, prompt, and calculator-observation tokens. GRPO stores old-policy and frozen-reference log probabilities from the actual temperature-scaled rollout distribution, then recomputes only policy-generated tokens with gradients. The agent stage uses the same loss over complete two-turn episodes, sampling one easy, medium, and hard prompt per update at temperature 0.9 with KL coefficient 0.1. Prompt groups are accumulated separately to keep peak memory close to a one-prompt update.

## Reading the results

Accuracy is not assumed to rise monotonically. `metrics.jsonl` records per-update and aggregate training metrics, direct and tool-enabled accuracy, behavior by difficulty, protocol validity, preference margin, sampled KL, tool attempts, valid executions, unnecessary tool use, and rollout diagnostics. `train_<stage>_trajectories.jsonl` saves the lowest- and highest-reward sample for every prompt group; `eval_<stage>.jsonl` keeps every evaluation generation. Agent RL stops before saving a checkpoint if its first five updates execute no valid calculator call.

The CLI also reports a 0–100 **quality index**. It first combines direct and tool-enabled task success as `0.4 * direct + 0.6 * tool`, then takes a weighted geometric mean of task success (55%), valid formatting (20%), held-out preference accuracy (15%), and tool efficiency `1 - unnecessary_tool_rate` (10%). The geometric mean makes a collapse in one dimension difficult to hide behind a strong score elsewhere. The index is a compact summary, not a training objective; its component metrics and raw generations remain the authoritative diagnosis. JSON stores the index on the same 0–1 scale as the component rates. Historical evaluation records are upgraded in memory when plotted, so existing runs do not need to be repeated.

The SVG plots the index and its component behaviors together so an always-tool policy cannot masquerade as an unqualified capability gain.

## Safety boundary

Calculator expressions are parsed with an allowlisted Python AST walker. Model output is never passed to `eval`, `exec`, a shell, or another general-purpose interpreter.

See [plan.md](plan.md) for the complete design, capability gates, and implementation rationale.
