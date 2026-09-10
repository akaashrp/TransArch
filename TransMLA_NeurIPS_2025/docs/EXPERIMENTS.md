# TransMLA quality experiments

Run from `TransMLA_NeurIPS_2025/` on branch `qwen3-mimo-conversion`.
This campaign measures quality retention for Qwen3-4B and MiMo-7B-RL-0530.
GPU validation and experiments are prepared as jobs; preparation submits none.

The [initial preparation record](../experiments/IMPLEMENTATION_VALIDATION.json) contains
the 31-test CPU suite result, four final collector checks, seven verified weight
shards, both offline preflights and launcher previews, and eight successful
Slurm `--test-only` checks. The [GSM8K update record](../experiments/GSM8K_NO_THINK_VALIDATION.json)
documents the no-thinking prompt checks and refreshed launch bundles.
Full-size GPU validation and quality results are pending.

## Launch the prepared campaign

The primary bundle is `experiments/prepared/main/`. The alternative bundle
`experiments/prepared/with-milder/` additionally includes rank 1024 for each
family. Choose one bundle: running both would duplicate teacher/rank-512 work.

Preview the exact submission chain, including a fresh CPU artifact check:

```bash
bash experiments/prepared/main/submit.sh
```

When ready to allocate GPUs and run the experiments:

```bash
bash experiments/prepared/main/submit.sh --execute
```

Only the explicit `--execute` path calls `sbatch`. There are no watchers,
scheduled submissions, held jobs, or background launch agents. The launcher
writes `submission.json` after each successful submission and refuses a second
submission of the same bundle. Inspect that journal if submission is interrupted.

Each stage is a Slurm array with `afterok` dependencies on the preceding array:

| Stage | Primary array tasks | Work |
| --- | ---: | --- |
| source | 2 | Full source weights, cached/padded logits, native MiMo decoder parity, HF generation |
| convert | 2 | Calibration, stage perplexity diagnostics, export, reload parity, 4K–32K GPU inference |
| diagnostic | 4 | Two PIQA, two MATH-500, two 4K NIAH-multikey items per model row |
| full | 476 | Commonsense, GSM8K, math and retrieval shards |

The primary rows are Qwen3 teacher, Qwen3 TransMLA rank 512, MiMo teacher,
and MiMo TransMLA rank 512. The alternative has six rows, four conversions,
six diagnostic tasks, and 714 full-evaluation tasks.

Every array is capped at two concurrent tasks. Each task requests one H100
80 GB, eight CPU cores, 128 GB host RAM, partition `GPU-shared`, account
`cis260115p`, QoS `gpu`. Time limits are 2 hours for source/diagnostic,
8 hours for conversion, and 24 hours per full-evaluation shard. These are
resource requests, not measured runtime estimates. A failed GPU validation
prevents dependent stages from starting. Workers also verify their prerequisite
validation reports, checkpoint identity, code and environment.

No full-model GPU result is implied by a successful CPU preflight. The full
weight and long-context checks run in the first two stages after submission.

## Frozen protocol

Source snapshots:

| Family | Source | Revision |
| --- | --- | --- |
| Qwen3 | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| MiMo | XiaomiMiMo/MiMo-7B-RL-0530 | 323400599af3903adc2a536d6340a23fee88d2e0 |

Conversion uses rank 512 (optionally also 1024), RoPE dimension 64, folding
factor 4, balanced KV ratio 1, and no query compression. Calibration is
128 sequences capped at 256 tokens from the pinned WikiText-2 **train** split,
seed 42, batch size 4. Eight separate validation sequences, seed 43, measure
perplexity before positional reduction, after partial RoPE, and after PCA.
Validation does not tune the configuration. No optimizer or recovery training
runs. The export records the actual calibration sample/token count and hash.
CPU tokenization preflight produced 128 samples and 32,412 real tokens for
each family; tokenization/round-tripping can yield fewer than 256 per sample.

For both source architectures, rank 512 retains 28.125% of the original
logical KV elements per token; rank 1024 retains 53.125%. These fractions
describe cache tensors, not total GPU memory or serving speed. Qwen3 is
explicitly the adaptation preserving its nonlinear Q/K RMSNorm.

The unchanged prompt/scoring functions are extracted from Linearization commit
`4f59436c2438f8d339183c19bbcd953ea0b598b6`; their provenance is in
`experiments/protocol_source.json`. CPU tests verify the extraction against the
source checkout. Evaluation runs independently of that checkout.

| Task | Protocol and sharding |
| --- | --- |
| PIQA, HellaSwag, ARC-Easy/Challenge, Winogrande | lm-eval 0.4.12, zero-shot, acc_norm where defined |
| MMLU | lm-eval 0.4.12, five-shot, accuracy |
| GSM8K | All 1,319 test examples, five-shot, native no-thinking chat prompt, greedy, 1,024 output tokens, strict/flexible extraction |
| MATH-500 thinking | 500 problems, one sample, T=0.6/p=0.95/k=20, 32,768 output tokens; 25 problems per shard |
| MATH-500 no-thinking | Same 500, one sample, T=0.7/p=0.8/k=20, 4,096 output tokens; 25 per shard |
| AIME24/25 | 30 each, eight samples per problem, thinking, T=0.6/p=0.95/k=20, 32,768 output tokens; five problems per shard |
| NIAH single/multikey/multiquery | 4K/8K/16K/32K; five independent seed shards of 100 items per cell, seeds 0–4; greedy, thinking off |

NIAH uses the pinned Qwen3 tokenizer to construct item text across both
families. Every row gets the same underlying items and the same **128-token
answer cap**, with no answer prefill. Historical MiMo cells sometimes used
larger caps; those cells are not directly comparable. Truncation and output
length are retained so cap-related behavior is visible.

The HF runner uses native chat templates and EOS settings, explicit sampling,
and a stable seed per math problem. All eight AIME samples for a problem are
generated together. Sampling trajectories are not expected to match vLLM.
Teacher controls are rerun in the same HF path. Harness few-shot construction
keeps seed 1234, matching the shared runner's defaults. Likelihood tasks use
the original plain harness prompts. GSM8K places the unchanged five-shot
question/answer text in one user message and applies the native chat template
with `enable_thinking=False` for every teacher and TransMLA row. Its greedy
decoding, 1,024-token output cap, stop strings and strict/flexible extraction
remain the harness protocol. Earlier plain-prompt GSM8K results are a different
protocol and cannot be resumed or mixed with these runs.

All requested data is staged under `.cache/experiment-data/`, including the
63 leaf tasks behind the seven harness entries. Dataset revisions, split
fingerprints, row counts and local file SHA256s are recorded. Compute jobs
operate with Hugging Face network access disabled. The offline dataset adapter
keeps the upstream harness task definitions and replaces only dataset loading.

## Attention implementation and validation

The portable model keeps latent KV and decoupled RoPE keys in DynamicCache.
For eligible CUDA prefill, it temporarily reconstructs per-head K/V and pads
the value width to use Torch Flash SDPA at head width 192. Decode and masked
or unsupported prefill use tiled latent attention. Score tiles are limited
to 64 query positions and a configurable 128 MiB FP32 score budget; no full
sequence-by-sequence mask is created by the model. The total operation also
uses linear-sized projections/cache, and temporary softmax buffers.

The GPU conversion gate checks fused/latent parity, saved/reloaded logits,
cached and padded decoding, actual latent cache dimensions, and prefill plus
decode at 4K/8K/16K/32K. It requires the fused prefill path on the requested
H100. Failure is reported before a full quality run is released.

```bash
bash scripts/test_experiments.sh
```

This runs tiny CPU models, actual HF generation and lm-eval integration, real
attention dimensions, native MiMo decoder parity using pinned remote source,
mask/chunk/cache checks, protocol extraction, resumability, metrics, and job
preparation tests. The CPU suite is implementation evidence only.

## Results, resuming, and external comparisons

Math and NIAH write one atomic JSON file per completed problem/item, including
raw output text, token IDs, input/prompt hashes, seed, correctness, token count,
finish reason and truncation. An interrupted problem is rerun; completed ones
are reused only with the same model/protocol/code/environment identity. Harness
tasks save their raw metrics and samples at task completion; an interrupted
harness task must restart.

Collect at any point:

```bash
source scripts/eval_env.sh
"$TRANSMLA_PYTHON" -m transmla.experiments.collect \
  --plan experiments/prepared/main/plan.json \
  --out experiments/prepared/main/summary
```

The collector reports missing shard indices, partial sample counts, pass@1
and pass@8, retrieval confidence intervals, truncation, output lengths, and
teacher deltas only for complete matched cells. Missing work is never zero.
It rejects changed evaluation arguments and mixed checkpoints within a model row.
The JSON contains all metrics; the Markdown table presents primary accuracy.

After inspecting and correcting a failure, resubmit only the needed array
indices, e.g. `sbatch --array=7,12%2 experiments/prepared/main/full.sbatch`.
This is an explicit submission command. A failed predecessor may leave later
arrays pending with `DependencyNeverSatisfied`; inspect the saved job IDs
before canceling or replacing those exact arrays. Do not launch a second
whole chain blindly. A changed model/protocol/code requires a new bundle and
fresh GPU validation. An interrupted conversion retains its `.incomplete`
directory for inspection; it is never automatically deleted or overwritten.

SWA and the chosen GDN+OPD endpoint are external comparison inputs, not new
training jobs. `experiments/comparison_inputs.example.json` lists the required
checkpoint/artifact, source revision, window/layer settings, training budget,
and protocol metadata. Anthony's SWA handoff and the selected paper endpoint
are still needed. Historical tables alone do not establish matched prompts,
caps, checkpoint provenance, or backend comparability; keep those rows pending
until the raw artifacts can be audited.

## Recreate the preparation

```bash
bash scripts/setup_eval_env.sh
bash scripts/prepare_experiments.sh --out experiments/prepared/new-main
# Alternative including the milder compression point:
bash scripts/prepare_experiments.sh --out experiments/prepared/new-with-milder --include-milder
```

The evaluation venv inherits the existing conda Torch installation and installs
its own pinned evaluation dependencies; it does not modify the conda environment.
`requirements-evaluation.txt` pins the tested core versions, and
`.cache/evaluation-environment.txt` records the full resolved environment.
Runtime temporary files and dynamic model-code caches stay inside this checkout.
The staging script uses standard HTTP downloads; the Xet transfer path stalled
on this shared filesystem during preparation. Partial files from that attempt
are retained separately, and only verified/download-complete blobs are used.

An optional full-size FP32 CPU loading probe exited with code 137 during
checkpoint loading. It is not counted as a passing source-model test.
Full-size numerical validation remains in the explicitly provisioned GPU stages.
