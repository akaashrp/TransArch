# Qwen3 and MiMo conversion implementation

The `qwen3-mimo-conversion` branch is based on upstream commit
`eaf4dde7a9b0592566166fd347959738e70ecefd`. The upstream TransMLA URL now redirects
to MuLabPKU/TransArch; the relevant project is `TransMLA_NeurIPS_2025/`.

This work implements training-free conversion and portable Transformers
inference. Experiment selection and quality benchmark runs are separate work.
No optimizer, distillation, or recovery training is invoked by this entry point.

## Supported sources and fidelity

| Source | Conversion path | Export |
| --- | --- | --- |
| `XiaomiMiMo/MiMo-7B-RL-0530` | Upstream RoRoPE/FreqFold and balanced joint KV PCA, with QKV bias preserved | Portable `TransMLAForCausalLM` |
| `Qwen/Qwen3-4B` | Same rotation and PCA stages, applied after the source Q/K normalization | Portable `TransMLAForCausalLM`; norm-preserving Qwen3 adaptation |
| Qwen2 dense full-attention checkpoints | Same linear path as MiMo | Portable `TransMLAForCausalLM` |

MiMo's ordinary decoder loads through the Qwen2 implementation. Its unused MTP
branch is omitted and reported; missing decoder weights or unrecognized extra
weights are errors. The tokenizer, chat template, generation configuration,
native RoPE settings, vocabulary, and tied embeddings are preserved.

Qwen3's per-head normalization is nonlinear. Applying TransMLA directly to the
unnormalized projection weights would change the source function before the
intended approximation even starts. This implementation preserves both source
norms, rotates normalized keys, and compresses the joint normalized-key/value
activations. It must be reported as a **Qwen3 adaptation**, not as an unmodified
upstream DeepSeek-format conversion. Query LoRA and new latent normalization
are intentionally unavailable in this path.

The portable attention backend uses PyTorch eager/SDPA and stores only
`kv_lora_rank + qk_mqa_dim` elements per layer per cached token, in DynamicCache.
It absorbs the non-RoPE key up-projection into queries and the value
up-projection into the attention output. It retains the native query scaling.
This backend supports logits and autoregressive generation; it is not a vLLM
plugin, and a stock DeepSeek loader cannot interpret the Qwen3 adaptation.
Static/quantized caches, sliding-window sources, extended-context RoPE recipes,
and distributed conversion are outside the supported path.

## Entry points

Run from `TransMLA_NeurIPS_2025/`. Use `requirements-conversion.txt` for this
path, rather than upstream's older vLLM requirements. On the shared workspace,
activate `conda activate vllm` first; this implementation does not import vLLM.

Inspect the native source configuration without loading weights or data:

```bash
bash scripts/convert_qwen3.sh --inspect
bash scripts/convert_mimo_rl.sh --inspect
python -m transmla.convert_pretrained --help
```

The inspection output uses upstream-style defaults as a configuration check;
it does not select an experiment. For a later conversion, supply the chosen
revision, compression settings, and calibration source explicitly:

```bash
bash scripts/convert_qwen3.sh \
  --revision "$TRANSMLA_REVISION" \
  --save-path "$TRANSMLA_OUTPUT" \
  --calibration-file "$TRANSMLA_CALIBRATION" \
  --kv-lora-rank "$TRANSMLA_KV_RANK" \
  --qk-mqa-dim "$TRANSMLA_ROPE_DIM" \
  --freqfold "$TRANSMLA_FREQFOLD" \
  --device cuda:0 --dtype bf16
```

Use `convert_mimo_rl.sh` for MiMo, or set `TRANSMLA_SOURCE_MODEL` to a local
source checkpoint and add `--local-files-only`. Calibration JSONL contains one
`{"text": "..."}` object per line. The alternative `--cal-dataset` option reads
only the dataset's training split. This entry point never searches a test set
for the best folding factor. Output directories must be new or empty.

The export contains weights, a self-contained Python model/config file,
tokenizer and generation configuration, and `conversion_report.json`. The
report records source identity/revision, arguments, calibration fingerprint or
file hash, actual sample/token counts, software versions, code hash, ignored
MTP tensors, zero optimizer steps, and whether the norm-preserving adaptation
was used. The rank determines the logical cache width, not a measured quality
or speed result.

Load an export without importing this checkout:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    output_path, trust_remote_code=True, local_files_only=True,
    torch_dtype="auto", attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained(output_path, local_files_only=True)
```

## Implementation verification

```bash
bash scripts/test_conversion.sh
```

The tests use tiny random CPU models and local calibration text, with network
access disabled. All temporary files and dynamic model-code caches stay in the
project's `.cache/` directory. Numerical checks cover:

- RoRoPE equivalence when all positional components are retained, including
  Qwen3's nonuniform Q/K norm weights and its unequal hidden/query widths.
- Full-rank MLA equivalence to the partial-RoPE reference.
- The absorbed linear path against upstream attention, including query LoRA.
- Preservation of non-attention weights, norms, tied embeddings and tokenizer.
- Self-contained save/reload, causal and padded attention, cached multi-token
  decoding, greedy generation, and actual latent cache tensor dimensions.
- Input validation, output protection, hook cleanup on calibration failure,
  and perplexity masking when padding and EOS use the same token ID.

Validation on 2026-09-07: **14 CPU tests passed** with Torch `2.8.0+cu129`,
Transformers `4.56.1`, and pytest `9.0.2`, including FP32 and BF16 conversion.
The native source configurations and tensor-name inventories were also checked
without loading model weights:

| Source revision | Expected decoder tensors | Allowed MTP tensors | Missing/unrecognized |
| --- | ---: | ---: | --- |
| Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c` | 399 | 0 | 0 / 0 |
| MiMo-7B-RL-0530 `323400599af3903adc2a536d6340a23fee88d2e0` | 435 | 16 | 0 / 0 |

These checks establish implementation behavior on tiny models and compatibility
of the source configurations/tensor names. They are not full-size conversion,
GPU execution, or benchmark results.

Upstream fixes shared with the older converter include modern attention/cache
arguments, correct query projection width, explicit K/V expansion for SDPA,
calibration after Q/K norms, removal of hooks on errors, and KV balancing over
all value channels (values have no separate RoPE component). PCA/RoRoPE remain
the upstream algorithms. The legacy `converter.py`/DeepSeek export path remains
available but is not the validated entry point for these source models.
