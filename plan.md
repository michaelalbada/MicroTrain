# microTrain implementation plan

## Thesis

`microTrain` explains how a pretrained base model becomes useful through a modern post-training pipeline:

```text
SmolLM2-135M or 360M base
        |
        v
       SFT             learn the response and tool protocols
        |
        v
       DPO             learn offline preferences
        |
        v
   GRPO / RLVR         improve verified single-turn correctness
        |
        v
   tool-use RL         learn multi-turn calculator interaction
        |
        v
   shared evaluation   measure the capability change after every stage
```

The project is an executable curriculum, not a general training framework. Every algorithm should be readable from data to loss to optimizer step without tracing through trainer abstractions.

## Goals

- Start from the public `HuggingFaceTB/SmolLM2-135M` **base** checkpoint by default, with `HuggingFaceTB/SmolLM2-360M` as a larger preset; never use the instruction-tuned variants.
- Implement the model forward pass and all training algorithms directly in PyTorch.
- Use one deterministic arithmetic dataset and calculator environment throughout the curriculum.
- Full-finetune the model so the algorithms remain explicit; do not introduce LoRA or PEFT in the primary path.
- Evaluate the same held-out task suite after the base, SFT, DPO, RLVR, and tool-use RL stages.
- Produce measured capability curves from saved evaluation results.
- Save checkpoints in a stable format that a later `microServe` project can load.
- Keep each algorithm implementation small enough to understand in one sitting.

## Non-goals

- General-purpose model, dataset, reward, or trainer plugins.
- Distributed training, tensor parallelism, FSDP, DeepSpeed, or custom CUDA kernels.
- Compatibility with arbitrary Hugging Face architectures.
- Matching TRL, VERL, or production RL infrastructure throughput.
- Teaching pretraining; pretrained base weights are the starting point.
- Building the serving runtime in this repository.
- Claiming broad mathematical reasoning or agent capability from a narrow arithmetic environment.

## Design principles

1. **Capability progression over algorithm collection.** Each stage has a distinct behavioral purpose and is evaluated before the next stage begins.
2. **Equations map directly to code.** Core losses should be recognizable in roughly 20-40 lines.
3. **No generic trainer.** Repetition between training loops is preferable to an abstraction that hides algorithm-specific mechanics.
4. **One semantic environment.** SFT demonstrations, preference pairs, verifier prompts, tool trajectories, and evaluation examples are generated from the same task definition.
5. **Measured claims only.** README plots and tables are generated from evaluation artifacts, never illustrative hard-coded percentages.
6. **HF at the boundary.** Hugging Face is used to acquire weights and tokenizer artifacts, not to implement training or generation.
7. **Ablations remain possible.** The primary path is cumulative, but stages can be run independently from their declared input checkpoint.

## Proposed repository layout

```text
microtrain/
|-- model.py          # handwritten SmolLM2-compatible transformer
|-- tokenizer.py      # tokenizer.json wrapper and prompt formatting
|-- checkpoint.py     # download, load, validate, and save checkpoints
|-- data.py           # deterministic data generation and split manifests
|-- environment.py    # arithmetic tasks and safe calculator tool
|-- rollout.py        # batched single-turn and multi-turn generation
|-- rewards.py        # verifiers, preference construction, episode rewards
|-- sft.py            # supervised fine-tuning
|-- dpo.py            # direct preference optimization
|-- grpo.py           # single-turn and trajectory-level GRPO
|-- eval.py           # common evaluation suite and result writer
|-- plot.py           # capability-curve generation from measured results
|-- run.py            # stage-oriented command-line entry point
`-- __init__.py
tests/
|-- test_model.py
|-- test_losses.py
|-- test_environment.py
|-- test_rollout.py
`-- test_checkpoint.py
README.md
pyproject.toml
```

Generated files live outside the Python package:

```text
runs/<run-name>/
|-- config.json
|-- data_manifest.json
|-- metrics.jsonl
|-- capability_curve.svg
`-- checkpoints/
    |-- base/
    |-- sft/
    |-- dpo/
    |-- rlvr/
    `-- agent_rl/
```

## Dependencies

Runtime dependencies should be limited to:

- `torch` for model execution, optimization, and sampling.
- `huggingface_hub` for downloading the public base checkpoint.
- `safetensors` for loading and saving model weights.
- `tokenizers` for the released SmolLM2 tokenizer.
- `rich` for progress, compact metric tables, and artifact summaries in the CLI.

Testing may use `pytest`. Plotting should either produce a small SVG directly or use an optional plotting dependency; plotting libraries must not become training dependencies.

`uv` owns environment creation, dependency resolution, and the committed lockfile. `uv sync` installs the default development group; `uv sync --no-dev` installs only runtime dependencies.

`transformers` is an optional development dependency used only for logit-parity validation against the upstream model. It must not be imported by the runtime package.

## Model and checkpoint contract

### Upstream model

- Default model: `HuggingFaceTB/SmolLM2-135M` at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Larger preset: `HuggingFaceTB/SmolLM2-360M` at revision `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`
- Starting stage: pretrained base
- License: Apache 2.0
- Architecture family: Llama-compatible decoder-only transformer
- 135M configuration: 30 layers, hidden size 576, 9 query heads, and 3 key/value heads.
- 360M configuration: 32 layers, hidden size 960, 15 query heads, and 5 key/value heads.
- Shared configuration: vocabulary size 49,152 and maximum context length 8,192.

The first implementation can train with a much shorter context, such as 256 tokens, without changing model weights or architecture.

### `model.py`

Implement only the features needed by this checkpoint:

- Token embeddings with tied language-model output weights.
- RMSNorm.
- Rotary position embeddings.
- Grouped-query causal self-attention.
- SwiGLU MLP.
- Full-sequence training forward pass.
- Optional incremental KV-cache interface needed to validate the future serving contract.

Use `torch.nn.functional.scaled_dot_product_attention` where it keeps the implementation clear. Avoid configuration branches for unsupported architectures.

The model API should expose explicit tensors rather than Hugging Face output classes:

```python
logits = model(input_ids, position_ids=None)
logits, cache = model.forward_cached(input_ids, cache, position_ids)
```

### Checkpoint format

Each stage checkpoint should remain structurally compatible with the base model:

```text
checkpoint/
|-- config.json
|-- tokenizer.json
|-- tokenizer_config.json
|-- special_tokens_map.json   # only if present upstream
|-- model.safetensors
`-- microtrain.json           # stage, parent, data hash, seed, metrics
```

Do not add calculator-specific vocabulary tokens in the first version. Express tool calls with ordinary text tokens so the vocabulary and tied embedding shapes remain unchanged.

### Model validation gate

Before implementing post-training, verify:

- All expected upstream tensors load with no ignored or missing parameters.
- Greedy token generation is deterministic for a fixed prompt and seed.
- Full-sequence and cached decoding logits agree within tolerance.
- Optional development test: handwritten-model logits match `transformers` logits within a documented numerical tolerance.

## Dataset and environment

### Task family

Use procedurally generated integer arithmetic expressions with controlled difficulty:

- Easy: one operation with small operands; the model should often answer directly.
- Medium: multiple operations or larger operands; either direct reasoning or a tool is viable.
- Hard: large or nested expressions designed to benefit strongly from the calculator.

Example episode:

```text
User: What is (37 * 48) + 19?
Assistant: <tool>{"expression":"37 * 48 + 19"}</tool>
Tool: <result>1795</result>
Assistant: <answer>1795</answer>
```

The exact surface protocol will be finalized before dataset generation and then treated as stable.

### Safe calculator

The calculator must parse an allowlisted expression grammar. Never pass model output to Python `eval` or a shell. Initially support:

- Integer literals.
- Parentheses.
- Addition, subtraction, multiplication, and exact or explicitly defined integer division.
- Unary negation if needed by the generated task distribution.

Reject unknown syntax, excessive expression size, divide-by-zero, and values outside configured bounds.

### One environment, multiple views

The seeded task generator produces stable manifests for:

- Base evaluation prompts.
- SFT prompt/completion demonstrations.
- DPO chosen/rejected response pairs.
- RLVR training prompts with hidden verifier answers.
- Agent-RL prompts with calculator access.
- Held-out evaluation prompts.

The current curriculum uses all easy training operators for the single-turn RLVR view. The Agent-RL view is an interleaved, near-equal mix of easy tasks where direct answers avoid the tool cost, medium tasks near the capability frontier, and hard tasks where calculator use is strongly beneficial.

The split must be structural, not merely a random split of serialized examples. Hold out combinations of operand ranges, operators, expression templates, or depths so evaluation measures generalization rather than exact-example recall.

### Avoiding trivial policies

The environment must make tool-use decisions meaningful:

- Include easy examples where direct answers are efficient.
- Include hard examples where calculator use is reliably beneficial.
- Penalize malformed calls and invalid expressions.
- Apply a small tool-use or step cost.
- Do not reward hidden reasoning text; reward observable correctness and protocol behavior.
- Include adversarial evaluation cases that detect answer leakage, parser exploits, multiple-answer ambiguity, and reward hacking.

## Training stages

### Stage 0: base evaluation

Evaluate the unmodified base checkpoint before any training. This establishes that later gains came from the curriculum and helps calibrate task difficulty.

Success criteria:

- The base produces valid language but has meaningful room to improve.
- The task is neither at zero signal nor near saturation.
- Direct-answer, formatting, and tool-use metrics are recorded separately.

If the base is at floor or ceiling, adjust the environment before creating permanent training splits.

### Stage 1: supervised fine-tuning

Teach the response grammar and representative task demonstrations.

Loss:

```text
L_sft = -mean(log p(completion token | prompt, preceding completion tokens))
```

Requirements:

- Mask prompt and padding tokens from the loss.
- Mix direct-answer and calculator-trajectory demonstrations.
- Keep the training loop explicit: sample batch, forward, masked cross-entropy, backward, update.
- Save the SFT checkpoint used as the DPO reference.

Expected capability change:

- High answer-tag and tool-call syntax validity.
- Improved easy arithmetic.
- Imitation of demonstrated tool behavior, without yet requiring optimal tool choice.

### Stage 2: direct preference optimization

Teach offline response preferences from paired completions. Construct correctness and formatting negatives with the same strategy and approximate length as chosen answers to prevent superficial shortcuts. Explicit strategy pairs compare a direct response with a tool trajectory and therefore differ in length by design; the frozen-reference ratio remains essential for those pairs.

Loss:

```text
margin = (
    logp_policy_chosen - logp_reference_chosen
    - logp_policy_rejected + logp_reference_rejected
)
loss = -logsigmoid(beta * margin).mean()
```

Requirements:

- Freeze the SFT reference policy.
- Compute log probabilities only over response tokens.
- Precompute reference log probabilities for the fixed preference dataset.
- Include preference pairs for correctness, formatting, malformed tool calls, and unnecessary tool use.

Expected capability change:

- Higher held-out preference accuracy.
- Better correctness and protocol adherence without online sampling.

### Stage 3: single-turn GRPO / RLVR

Improve final-answer correctness with an exact verifier and on-policy samples.

Conceptual update:

```python
responses, old_logprobs = sample(policy, prompts, group_size=G)
rewards = verifier(prompts, responses)
advantages = group_normalize(rewards)

logprobs = policy.logprobs(prompts, responses)
loss = clipped_policy_loss(logprobs, old_logprobs, advantages)
loss += beta * reference_kl(policy, reference, prompts, responses)
```

Requirements:

- Generate rollouts without autograd, then recompute response log probabilities with gradients.
- Cache old-policy and reference-policy log probabilities for each rollout batch.
- Normalize rewards independently within each prompt group.
- Produce zero advantages for zero-variance groups.
- Use response-token masks consistently for policy and KL losses.
- Keep the DPO checkpoint frozen as the stage reference unless experiments show a clearer alternative.

Expected capability change:

- Improved exact-match accuracy on verified single-turn tasks.
- No major collapse in formatting, length, or output diversity.

### Stage 4: tool-use agent RL

Extend the same GRPO machinery from single responses to complete multi-turn episodes:

```text
prompt -> assistant action -> optional tool result -> assistant action -> final answer
```

The environment, not the model, executes the calculator and appends tool results. Episode reward is assigned from observable outcomes:

```text
reward = answer_correct
       + valid_protocol_bonus
       - invalid_action_penalty
       - tool_call_cost
       - excess_step_cost
```

Requirements:

- Record token log probabilities only for policy-generated assistant tokens, never user or tool-result tokens.
- Bound the number of tool calls, turns, tokens, and expression complexity.
- Normalize rewards across groups of episodes for the same initial prompt.
- Use a balanced easy/medium/hard prompt batch for every optimizer update.
- Accumulate prompt-group gradients separately so the balanced update does not triple peak activation memory.
- Record attempted tool calls separately from valid calculator executions.
- Save representative lowest/highest-reward trajectories for every prompt group.
- Stop before checkpointing if the first five updates execute no valid calculator call.
- Make the stage optimizer reuse the same visible GRPO loss rather than introducing a separate agent framework.
- Freeze the RLVR checkpoint as the KL reference for this stage.

Expected capability change:

- Higher accuracy on tool-beneficial expressions at the calibrated frontier.
- Increased tool use where beneficial.
- Low malformed-call and unnecessary-tool rates.

## Evaluation suite

Every stage runs the identical deterministic evaluation manifest. Record both aggregate metrics and per-example outputs.

Primary metrics:

- Aggregate quality index: a 0–1 weighted geometric mean of task success (55%), valid formatting (20%), preference accuracy (15%), and tool efficiency (10%). Task success is `0.4 * direct accuracy + 0.6 * tool accuracy`; tool efficiency is `1 - unnecessary tool rate`. Display it as 0–100, but retain every component metric for diagnosis.
- Overall exact-answer accuracy.
- Accuracy by easy, medium, and hard difficulty.
- Direct-answer accuracy with tools disabled.
- Tool-enabled episode success rate.
- Valid response-format rate.
- Tool-attempt rate.
- Valid tool-call rate.
- Tool-use rate by difficulty.
- Unnecessary-tool rate on easy tasks.
- Mean generated tokens, tool calls, and episode steps.
- Preference accuracy on held-out pairs.
- KL divergence from the stage's starting reference.

Integrity checks:

- Recompute answers independently from serialized training data.
- Include unseen expression templates and operand ranges.
- Test malformed and adversarial tool calls.
- Report confidence intervals or bootstrap intervals when the evaluation set is small.
- Save raw generations so improvements and regressions can be inspected.

The capability plot must be generated from `metrics.jsonl` and show the quality index alongside its component behaviors. The index is descriptive rather than an optimization target. Its geometric aggregation prevents a high score in one dimension from fully masking a collapse in another, while the component curves expose the reason for every change.

## Command-line experience

Target interface:

```bash
# Download and validate the public base checkpoint.
uv run microtrain prepare

# Calibrate and freeze a dataset/evaluation manifest.
uv run microtrain data

# Run one stage or the complete cumulative curriculum.
uv run microtrain sft
uv run microtrain dpo
uv run microtrain rlvr
uv run microtrain agent-rl
uv run microtrain all

# Re-evaluate saved checkpoints and regenerate the plot.
uv run microtrain eval
uv run microtrain plot
```

Commands should expose only the parameters readers are likely to vary. Algorithm constants remain together near the top of their implementation files, with their mathematical meaning documented.

Commands resolve prerequisites from checkpoint and manifest artifacts. Invoking a later stage announces and runs missing parents in curriculum order. Evaluation of a named stage may build that stage; plotting only backfills evaluations for checkpoints already present. Human-readable progress and compact capability tables are the default, with `--json` as the explicit machine-readable evaluation mode.

## Verification strategy

### Unit tests

- Tensor shapes and grouped-query attention expansion.
- RoPE and causal masking behavior.
- Completion-only loss masking.
- Sequence log-probability calculations.
- DPO loss on a hand-computed example.
- Group normalization, including zero variance.
- Clipped GRPO loss and detached old log probabilities.
- Calculator grammar and rejection paths.
- Episode masking: gradients apply only to assistant-generated tokens.
- Checkpoint round-trip and cached/full-forward parity.

### Tiny smoke tests

Each algorithm should support an intentionally tiny configuration and dataset that runs a few updates on CPU. Smoke tests verify execution and invariants, not capability improvements.

### Capability tests

The default SmolLM2-135M experiment should have declared gates rather than assumed monotonicity. Initial proposed gates:

- SFT materially improves protocol validity over base.
- DPO improves held-out preference accuracy over SFT.
- RLVR improves verified single-turn accuracy over DPO.
- Agent RL improves tool-enabled accuracy on the calibrated medium/hard frontier over RLVR.
- No stage introduces a large regression in format validity or easy-task accuracy without being reported.

Set numerical thresholds only after a pilot run establishes realistic variance. Do not tune against the final held-out manifest.

## Implementation phases

### Phase 1: foundation

1. Create packaging, dependency, and test skeletons.
2. Implement checkpoint download and tokenizer loading.
3. Implement the SmolLM2-compatible model.
4. Establish upstream logit parity and cached/full-forward parity.
5. Save and reload a byte-identical checkpoint round trip.

Exit gate: the handwritten model reproduces upstream inference for fixed inputs.

### Phase 2: environment and baseline

1. Implement the safe calculator and expression generator.
2. Define the stable conversation/tool protocol.
3. Generate provisional train, validation, and evaluation manifests.
4. Run base evaluation and inspect raw generations.
5. Adjust difficulty, then freeze version 1 of the manifests.

Exit gate: the base has nontrivial but clearly improvable performance, and the evaluation resists trivial always-tool or never-tool policies.

### Phase 3: SFT and DPO

1. Implement completion masking and SFT.
2. Train and evaluate the SFT checkpoint.
3. Generate hard preference pairs.
4. Implement DPO with precomputed reference log probabilities.
5. Train and evaluate the DPO checkpoint.

Exit gate: protocol validity improves after SFT and preference accuracy improves after DPO.

### Phase 4: RLVR

1. Implement batched grouped sampling.
2. Implement exact-answer rewards and group normalization.
3. Implement the clipped policy and reference-KL losses.
4. Add rollout diagnostics: reward variance, KL, entropy, lengths, and clipping fraction.
5. Train and evaluate the RLVR checkpoint.

Exit gate: verified single-turn accuracy improves without reward or format collapse.

### Phase 5: tool-use RL

1. Generalize rollout records from responses to episodes.
2. Execute and append calculator observations.
3. Implement policy-token masks and episode rewards.
4. Train trajectory-level GRPO.
5. Evaluate tool selection, correctness, cost, and failure modes.

Exit gate: hard-task success improves and the policy does not simply call the tool on every prompt.

### Phase 6: curriculum polish

1. Generate capability plots and representative trajectory comparisons.
2. Add optional ablations from SFT and random initialization.
3. Document expected memory, runtime, and numerical behavior by device.
4. Make every README result reproducible from a recorded command and run manifest.
5. Validate that the final checkpoint and tokenizer can be loaded by the future `microServe` implementation.

Exit gate: a new reader can reproduce the complete story and explain why each stage changed behavior.

## Risks and mitigations

### The model is too small for stable agent behavior

Keep the tool protocol short and regular, start with one tool and at most one call, and use curriculum difficulty. Compare the 135M and 360M presets without changing the teaching code or dataset.

### SFT saturates arithmetic accuracy

Increase held-out operand ranges or expression depth, and reserve harder structural templates for RLVR evaluation. Do not weaken SFT merely to make later stages look better.

### DPO duplicates the verifier objective

Use DPO primarily for offline behavioral preferences: formatting, concise responses, valid calls, and preference between plausible answers. Use RLVR for online exact correctness.

### GRPO receives mostly zero-variance groups

Calibrate sampling temperature and task difficulty, log zero-variance group frequency, and ensure prompts yield a useful mixture of successes and failures.

### The policy always calls the calculator

Include easy prompts, apply a small call cost, and report accuracy/cost Pareto curves rather than selecting a hidden reward weight solely for one headline number.

### Base-model contamination obscures learning

Measure the base checkpoint on the final suite, use procedurally generated held-out structures, and make claims about measured stage deltas rather than novelty relative to pretraining data.

### Readability erodes during optimization

Treat line count and indirection as review criteria. Performance optimizations belong only where profiling shows they are necessary for the default experiment.

## Version 1 defaults

- Protocol: `<answer>...</answer>` and `<tool>{"expression":"..."}</tool>`, with calculator observations injected by the environment.
- Data: 1,024 SFT examples, 1,024 preference pairs, 512 RLVR tasks, 512 agent-RL tasks, and 300 structurally held-out evaluations from seed 42.
- Context: natural variable-length batches with 32 generated tokens per action and at most two agent turns.
- SFT: 200 updates, batch size 8, learning rate `2e-5`.
- DPO: 100 updates, batch size 4, `beta=0.1`, learning rate `1e-6`.
- GRPO: 50 rollout batches, group size 8, two update epochs, clip range 0.2, KL coefficient 0.1, and temperature 1.2.
- Agent RL: the same GRPO implementation with three balanced prompts per update, temperature 0.9, KL coefficient 0.1, one allowed calculator call, a five-update valid-tool guardrail, and a 0.2 call cost.
- Model: SmolLM2-135M by default, with SmolLM2-360M selected by `--model 360m` and isolated in its own run directory.
- Runtime: CUDA, MPS, or CPU; automatic bfloat16 is limited to supported CUDA devices.

These are readable starting points, not claimed universal optima. Freeze the final evaluation manifest before publishing quantitative results, report all regressions, and calibrate on a separate development run rather than the final held-out records.

## Definition of done

Version 1 is complete when:

1. A user can download either supported SmolLM2 base preset and reproduce its logits with the handwritten model.
2. One command runs the cumulative SFT -> DPO -> RLVR -> tool-use RL curriculum.
3. Every algorithm uses direct PyTorch code with no Transformers, TRL, VERL, Axolotl, or generic trainer runtime.
4. Every stage produces a loadable checkpoint, raw evaluation records, and comparable metrics.
5. The generated capability plot shows measured behavioral changes and exposes tool-use costs and failure modes.
6. Tests cover the key mathematical invariants and environment safety boundaries.
7. The final checkpoint format is documented and suitable as input to `microServe`.
