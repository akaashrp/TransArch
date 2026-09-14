# TransMLA quality experiments

Run from `TransMLA_NeurIPS_2025/` on branch `qwen3-mimo-conversion`.
This campaign measures quality retention for Qwen3-4B and MiMo-7B-RL-0530.
GPU validation and experiments are prepared as jobs; preparation submits none.

The [initial preparation record](../experiments/IMPLEMENTATION_VALIDATION.json) contains
the 31-test CPU suite result, four final collector checks, seven verified weight
shards, both offline preflights and launcher previews, and eight successful
Slurm `--test-only` checks. The [GSM8K update record](../experiments/GSM8K_NO_THINK_VALIDATION.json)
documents the no-thinking prompt checks and refreshed launch bundles.
Current GPU job states are recorded in the prepared bundle's
`monitor/status.json`; the CPU validation records do not establish GPU success.

## Launch the prepared campaign

The current primary bundle is `experiments/prepared/main-score-fix-20260910/`.
`experiments/prepared/main/` retains the failed initial submission and its logs.
The optional rank-1024 bundle must be regenerated with `--include-milder` in a
fresh directory before use; its older prepared plan predates the validation fix.

Preview the exact submission chain, including a fresh CPU artifact check:

```bash
bash experiments/prepared/main-score-fix-20260910/submit.sh
```

When ready to allocate GPUs and run the experiments:

```bash
bash experiments/prepared/main-score-fix-20260910/submit.sh --execute
```

Only the explicit `--execute` path calls `sbatch`. The separate status monitor
described below does not submit jobs. The launcher
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

## Consolidated evaluation allocations

The original launcher maps each evaluation chunk to an individual Slurm task.
For an already validated campaign, `scripts/run_evaluation_pool.py` instead
distributes unfinished chunks across reusable GPU allocations. It preserves
the original plan, output paths, model code hash, numerical gates, seeds, and
sample coverage. Preparation validates all source/converted/diagnostic gates
and existing results before identifying unfinished work; it submits no jobs.

```bash
source scripts/eval_env.sh
"$TRANSMLA_PYTHON" scripts/run_evaluation_pool.py prepare \
  --plan experiments/prepared/main-score-fix-20260910/plan.json \
  --out experiments/prepared/main-score-fix-20260910/pool-new --workers 8
# Run this command once per allocated GPU, all pointing at the same pool:
"$TRANSMLA_PYTHON" scripts/run_evaluation_pool.py worker \
  --pool experiments/prepared/main-score-fix-20260910/pool-new/pool.json
```

Use a shared filesystem supporting `flock`. Each worker dynamically claims a
chunk, invokes the unchanged gated campaign worker, verifies the result, then
takes more work. The GPU allocation remains active across chunks. Evaluation
subprocesses and model loads remain separate to preserve the tested execution
path; this change removes per-chunk scheduler allocations, not all startup cost.
An interrupted chunk retains item checkpoints. A failed chunk is recorded and
not retried automatically; other available chunks continue. Inspect failures
and prepare a fresh pool for unfinished work after correcting their cause.
If an allocation times out, the monitor reports it; the pool never submits its
own replacement jobs. Do not run the old full array alongside the pool.

The monitor accepts a `stage_overrides.full` entry with `job_id`, `task_count`,
and `pool` (the pool directory). It tracks actual worker allocations separately
from `evaluation_progress` out of all 476 chunks. Original submission journals
remain unchanged. Pool manifests, per-chunk logs/states, and submission records
are stored alongside the original campaign, without modifying its frozen plan.

## Session monitor

The main campaign has a five-minute cron monitor installed on
`br012.ib.bridges2.psc.edu`. It calls `scripts/monitor_campaign.py` with
`experiments/prepared/main/monitor/config.json` and queues alerts through the
installed `codex queue` command to session
`01a07aaf-b487-7993-90ca-7acd747da7c4`.

Alerts cover stage starts and completion, failed/cancelled/timed-out/OOM tasks,
blocked dependencies, holds, and three consecutive scheduler/monitor errors.
Events are grouped per check and deduplicated. Slurm accounting supplies
completed tasks; the live queue takes precedence for running/requeued tasks.
An array is complete only when every expected member is terminal. Missing
queue entries alone do not establish success. No GPU is allocated by polling.

`monitor/status.json` records the task states, `monitor/health.json` the latest
poll health, and `monitor/alerts.jsonl` accepted alert deliveries. The
installation and initial delivery check are recorded in
`monitor/installation.json`. Alerts enter this session's native follow-up
queue; automatic processing requires its Codex client to remain running.
The cron watcher continues independently of the interactive shell.
Monitor configuration and state stay under `main/monitor/` across campaign
retries; the config's `bundle` field identifies the currently tracked submission.

The monitor reads `submission.json` each time, so replacing the journal with
a valid retry submission rearms it. When every tracked task is terminal it
stops querying Slurm until the journal changes. If recovery uses a different
prepared bundle, update `monitor/config.json` to its path. To pause alerts,
create `experiments/prepared/main/monitor/disabled`; remove that file to resume.
To uninstall, remove only the cron entry ending in
`# transmla-monitor-01a07aaf-b487-7993-90ca-7acd747da7c4` using `crontab -e` on
`br012`. Keep the monitor state and campaign artifacts for audit.

Preview a current snapshot without sending alerts:

```bash
python3 scripts/monitor_campaign.py --config experiments/prepared/main/monitor/config.json --preview
```

Monitor-only tests use synthetic Slurm snapshots and do not submit jobs:

```bash
python3 scripts/test_monitor_campaign.py
```

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
uses linear-sized projections/cache, and temporary softmax buffers. Latent
score products use FP32 operands with autocast disabled; casting an already
rounded BF16 score matrix to FP32 loses distinctions needed by softmax. The
temporary FP32 key copy is shared across heads; the stored cache stays BF16.

The GPU gate checks cached/padded logits and expanded-versus-latent prefill
algebra in FP32 with TF32 disabled and math SDPA (`atol=0.002`, `rtol=0.0001`).
The short export reference is computed after saving the BF16 weights, using
FP32 chunked attention both before save/reload comparison sides. Generation,
full evaluations, calibration, and exported weights remain BF16. The disposable
validation model is promoted to FP32 only after its BF16 runtime checks.

BF16 cached/padded/fused differences are recorded as maximum/RMS logit errors,
KL divergence and argmax disagreement counts; they are not asserted to be
elementwise equal. Each cached/padded/prefill comparison must have maximum
per-token KL at most 0.1; nonfinite outputs or KL also fail. This is a runtime
numerical gate, not a benchmark quality threshold. The BF16 gate also checks
native MiMo decoder parity, actual latent cache dimensions, HF generation, and
prefill plus decode at 4K/8K/16K/32K. It requires the fused prefill path on H100.

This replaces the original BF16 elementwise cache/padding assertion, which
failed on the unmodified source models. H100 diagnostic job `45678453` found
cache max errors of 0.500/0.422 in BF16 for Qwen3/MiMo, falling to
0.000077/0.000745 in FP32. FP32 padded max errors were 0.000321/0.001492;
native MiMo FP32 parity was exact. Fixed math SDPA and full BF16 reductions
did not remove the BF16 cached differences. Details are in
`experiments/recovery/cache-parity-20260910/diagnosis.json`.

H100 diagnosis `45707808` isolated an additional BF16 latent-score rounding
error: on the converted MiMo probe, FP32 score products reduced maximum KL
against the full FP32 reference from 1.2102 to 0.0118. The BF16 drift limit
above rejects the former path mismatch. See
`experiments/PREFILL_PRECISION_RECOVERY.json` for the controls and regression checks.

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
