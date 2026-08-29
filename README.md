# microTrain

Modern LLM post-training, stripped to the essentials.

`microTrain` starts with the public **SmolLM2-135M base** checkpoint and follows one model through a cumulative capability curriculum:

```text
base -> SFT -> DPO -> GRPO/RLVR -> tool-use RL -> evaluation
```

The same procedural arithmetic environment supplies demonstrations, preference pairs, verified rewards, calculator episodes, and held-out evaluation. The goal is to inspect what each stage changes—including regressions and degenerate policies—not to provide another extensible training framework.

## What is deliberately absent

- No Transformers model, generation, or Trainer APIs.
- No TRL, VERL, Axolotl, DeepSpeed, or PEFT.
- No generic trainer or algorithm registry.
- No arbitrary-model compatibility layer.

The runtime boundary is `torch`, `huggingface_hub`, `safetensors`, and `tokenizers`. The released Hugging Face checkpoint supplies weights and tokenizer artifacts; [`microtrain/model.py`](microtrain/model.py) implements the transformer itself.

The base is pinned to `HuggingFaceTB/SmolLM2-135M` revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. `prepare` downloads that exact public revision, loads every tensor into the handwritten model, and checks full-forward versus cached-decoding logits.

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
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first `prepare` needs Hugging Face network access. Later runs can reuse the local Hub cache. No Hugging Face token is required for this public checkpoint.

## Run

```bash
python -m microtrain.run prepare
python -m microtrain.run data
python -m microtrain.run sft
python -m microtrain.run dpo
python -m microtrain.run rlvr
python -m microtrain.run agent-rl
python -m microtrain.run eval --stage agent_rl
python -m microtrain.run plot
```

Or run the cumulative curriculum:

```bash
python -m microtrain.run all
```

Use `--smoke` for tiny manifests and one optimizer update per stage. Smoke mode proves execution and tensor invariants; it is not expected to produce capability gains.

The default run writes checkpoints, raw generations, metrics, and `capability_curve.svg` under `runs/default/`.

Useful development overrides are intentionally few:

```bash
python -m microtrain.run all --smoke --device cpu
python -m microtrain.run sft --steps 20
python -m microtrain.run eval --eval-limit 32   # all available checkpoints
```

Each run directory owns one immutable seeded manifest. Use a new `--run-dir` when changing the seed or smoke/full data size; this prevents accidentally comparing checkpoints trained on different data.

## Runtime expectations

SmolLM2-135M is small by current standards but this project full-finetunes it and keeps frozen references for DPO/GRPO. Allow several GB of memory. CPU is suitable for tests and smoke runs; a full curriculum is much more practical on CUDA or Apple Silicon. `auto` selects CUDA, then MPS, then CPU, and uses bfloat16 only when CUDA reports native support. Exact runtime depends heavily on the device, so measured artifacts—not hard-coded percentages or timing claims—are the source of truth.

For an optional upstream parity check:

```bash
pip install -e '.[dev]'
MICROTRAIN_BASE_CHECKPOINT=runs/default/checkpoints/base pytest tests/test_checkpoint.py -k parity
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

SFT and DPO mask user, prompt, and calculator-observation tokens. GRPO stores old-policy and frozen-reference log probabilities from the actual temperature-scaled rollout distribution, then recomputes only policy-generated tokens with gradients. The agent stage uses the same loss over complete two-turn episodes.

## Reading the results

Accuracy is not assumed to rise monotonically. `metrics.jsonl` records direct and tool-enabled accuracy, behavior by difficulty, protocol validity, preference margin, sampled KL, tool use, unnecessary tool use, and rollout diagnostics. `eval_<stage>.jsonl` keeps every raw generation. The SVG plots several metrics together so an always-tool policy cannot masquerade as an unqualified capability gain.

## Safety boundary

Calculator expressions are parsed with an allowlisted Python AST walker. Model output is never passed to `eval`, `exec`, a shell, or another general-purpose interpreter.

See [plan.md](plan.md) for the complete design, capability gates, and implementation rationale.
