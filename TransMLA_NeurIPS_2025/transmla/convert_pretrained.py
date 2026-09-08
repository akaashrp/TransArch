"""Training-free Qwen3/MiMo conversion, with a self-contained HF export.

Run from the project directory: python -m transmla.convert_pretrained --help
Calibration is the only data pass; this entry point does not run benchmarks,
select hyperparameters on a test set, or perform optimizer updates.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, Qwen2Config, Qwen3Config

from .lora_qkv import LoraQKV
from .modeling_transmla import TransMLAConfig, TransMLAAttention, TransMLAForCausalLM
from .partial_rope import PartialRope
from .qwen3_conversion import QKNormPartialRope, compress_qwen3_attention
from .utils import get_dataset, get_qkv_calibrate_outputs, prepare_dataloader


SUPPORTED_SOURCE_TYPES = ("qwen3", "mimo", "qwen2")


def source_config_from_dict(raw):
    """MiMo's ordinary decoder is a Qwen2 block; omit its unused MTP branch.

No remote Python code is needed. All ordinary-decoder weights must load, and
only explicitly identified MTP weights may be left unused.
"""
    source_type = raw.get("model_type")
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"Supported source model types are {SUPPORTED_SOURCE_TYPES}; got {source_type!r}")
    if raw.get("use_sliding_window") or "sliding_attention" in (raw.get("layer_types") or []):
        raise ValueError("Only full-attention source checkpoints are supported")
    if raw.get("rope_scaling") and raw["rope_scaling"].get("rope_type", raw["rope_scaling"].get("type", "default")) != "default":
        raise ValueError("Use the native source RoPE configuration; extended-context conversion is not validated")
    clean = {k: v for k, v in raw.items() if k not in ("model_type", "auto_map", "architectures")}
    cls = Qwen3Config if source_type == "qwen3" else Qwen2Config
    config = cls(**clean)
    config.head_dim = raw.get("head_dim") or config.hidden_size // config.num_attention_heads
    config._attn_implementation = "sdpa"
    return config


def make_export_config(source, *, kv_lora_rank, qk_mqa_dim, q_lora_rank=None,
                       source_model_type=None, o_proj_bias=False):
    raw = source.to_dict()
    source_type = source_model_type or source.model_type
    for key in ("model_type", "architectures", "auto_map", "_attn_implementation_autoset"):
        raw.pop(key, None)
    raw["head_dim"] = getattr(source, "head_dim", None) or source.hidden_size // source.num_attention_heads
    # Qwen2/MiMo can carry an inactive sliding_window value in their configs.
    raw["layer_types"] = ["full_attention"] * source.num_hidden_layers
    raw["sliding_window"] = None
    raw["use_sliding_window"] = False
    if source_type in ("mimo", "qwen2"):
        # HF Qwen2Attention always has Q/K/V bias, including when older
        # Qwen2 configs omit the attention_bias field altogether.
        raw["attention_bias"] = True
    raw.update(kv_lora_rank=kv_lora_rank, qk_mqa_dim=qk_mqa_dim,
               q_lora_rank=q_lora_rank, qk_norm_preserved=source_type == "qwen3",
               source_model_type=source_type, o_proj_bias=o_proj_bias)
    config = TransMLAConfig(**raw)
    config._attn_implementation = "sdpa"
    return config


def validate_conversion(config, freqfold, balance_kv_ratio):
    if freqfold <= 0 or freqfold % config.collapse or (config.head_dim // 2) % freqfold:
        raise ValueError("freqfold must be a positive multiple of collapse and divide head_dim/2")
    if balance_kv_ratio is not None and (not 0 < balance_kv_ratio < float("inf")):
        raise ValueError("balance_kv_ratio must be finite and positive")
    if config.q_lora_rank is not None and not 0 < config.q_lora_rank <= config.num_attention_heads * config.head_dim:
        raise ValueError("q_lora_rank must lie within the source query width")


@torch.no_grad()
def convert_model(model, calibration_batches, *, kv_lora_rank, qk_mqa_dim,
                  freqfold, q_lora_rank=None, balance_kv_ratio=1.0,
                  source_model_type=None, stage_callback=None):
    """Convert an already loaded source in place, then wrap its shared weights.

Using a meta-device wrapper avoids allocating a second full model. The returned
model is the canonical implementation for both export and cached inference.
"""
    source_config_from_dict({**model.config.to_dict(), "model_type": source_model_type or model.config.model_type})
    config = make_export_config(
        model.config, kv_lora_rank=kv_lora_rank, qk_mqa_dim=qk_mqa_dim,
        q_lora_rank=q_lora_rank, source_model_type=source_model_type,
        o_proj_bias=model.model.layers[0].self_attn.o_proj.bias is not None,
    )
    validate_conversion(config, freqfold, balance_kv_ratio)
    model.config.head_dim = config.head_dim
    model.eval().requires_grad_(False)
    batches = list(calibration_batches)
    if not batches or not any(int(batch["attention_mask"].sum()) for batch in batches):
        raise ValueError("Calibration requires at least one nonempty batch")
    for layer in model.model.layers:
        has_norm = hasattr(layer.self_attn, "q_norm") and hasattr(layer.self_attn, "k_norm")
        if has_norm != config.qk_norm_preserved:
            raise ValueError("Source attention normalization does not match the selected family")
    if stage_callback:
        stage_callback("source", model)
    activations = get_qkv_calibrate_outputs(model, batches, "Calibrating source Q/K/V")
    partial_cls = QKNormPartialRope if config.qk_norm_preserved else PartialRope
    for index, layer in enumerate(model.model.layers):
        layer.self_attn = partial_cls(layer.self_attn, activations["key"][index],
                                      freqfold=freqfold, collapse=config.collapse).eval()
    del activations
    if stage_callback:
        stage_callback("partial_rope", model)
    activations = get_qkv_calibrate_outputs(model, batches, "Calibrating partial-RoPE K/V")
    for index, layer in enumerate(model.model.layers):
        partial = layer.self_attn
        if config.qk_norm_preserved:
            converted = compress_qwen3_attention(
                partial, activations["key"][index], activations["value"][index], config, balance_kv_ratio,
            )
        else:
            upstream = LoraQKV(
                partial, activations["query"][index], activations["key"][index], activations["value"][index],
                q_lora_rank=q_lora_rank, qk_mqa_dim=qk_mqa_dim,
                collapse=config.collapse, kv_lora_rank=kv_lora_rank,
                balance_kv_ratio=balance_kv_ratio, rms_norm_eps=config.rms_norm_eps,
            )
            with torch.device("meta"):
                converted = TransMLAAttention(config, index)
            converted.load_state_dict(upstream.state_dict(), strict=True, assign=True)
        layer.self_attn = converted.eval()
    del activations
    with torch.device("meta"):
        exported = TransMLAForCausalLM(config)
    exported.load_state_dict(model.state_dict(), strict=True, assign=True)
    # RoPE frequencies are nonpersistent buffers, so load_state_dict does not
    # materialize them from the meta wrapper. Reuse the native source RoPE.
    exported.model.rotary_emb = model.model.rotary_emb
    exported.tie_weights()
    exported.generation_config = deepcopy(model.generation_config)
    exported.eval().requires_grad_(False)
    if stage_callback:
        stage_callback("converted", exported)
    return exported


def load_source_model(args, raw_config):
    config = source_config_from_dict(raw_config)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, revision=args.revision,
        local_files_only=args.local_files_only, torch_dtype=dtype,
        output_loading_info=True, attn_implementation="sdpa",
    )
    unexpected = info.get("unexpected_keys", [])
    mtp_prefixes = ("model.mtp_layers.",)
    allowed = [key for key in unexpected if raw_config["model_type"] == "mimo" and key.startswith(mtp_prefixes)]
    bad = [key for key in unexpected if key not in allowed]
    if info.get("missing_keys") or info.get("mismatched_keys") or bad:
        raise ValueError(f"Source decoder did not load exactly: {info}; unrecognized extra keys: {bad}")
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("Source loading left unmaterialized parameters")
    return model.to(args.device).eval(), allowed


def calibration_data(args, tokenizer):
    if args.calibration_file:
        import datasets
        path = Path(args.calibration_file)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or any(not isinstance(row.get("text"), str) for row in rows):
            raise ValueError("Calibration JSONL must contain objects with a text string")
        dataset = datasets.Dataset.from_dict({"text": [row["text"] for row in rows]})
        provenance = {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        dataset = get_dataset(args.cal_dataset)["train"]
        provenance = {"dataset": args.cal_dataset, "split": "train", "fingerprint": dataset._fingerprint}
    loader = prepare_dataloader(dataset, tokenizer, max_seqlen=args.cal_max_seqlen,
                                batch_size=args.cal_batch_size, nsamples=args.cal_nsamples, seed=args.seed)
    batches = [{k: v for k, v in batch.items() if k != "labels"} for batch in loader]
    provenance.update(samples=sum(b["input_ids"].shape[0] for b in batches),
                      tokens=sum(int(b["attention_mask"].sum()) for b in batches))
    return batches, provenance


def code_provenance():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"git_commit": proc.stdout.strip() if proc.returncode == 0 else None,
            "python_sources_sha256": digest.hexdigest()}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--save-path")
    parser.add_argument("--revision", help="Pin the source model and tokenizer revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Validate configuration without loading weights or data")
    parser.add_argument("--device", default="cpu", help="cpu or one CUDA device, e.g. cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--calibration-file", help="Local JSONL calibration data with a text field")
    data.add_argument("--cal-dataset", choices=("wikitext2", "ptb", "c4", "alpaca"), default="wikitext2")
    parser.add_argument("--cal-nsamples", type=int, default=128)
    parser.add_argument("--cal-max-seqlen", type=int, default=256)
    parser.add_argument("--cal-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freqfold", type=int, default=4, help="Fixed RoRoPE folding factor; no test-set search")
    parser.add_argument("--qk-mqa-dim", type=int, default=64)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--q-lora-rank", type=int, default=None, help="Optional query compression for MiMo/Qwen2 only")
    parser.add_argument("--balance-kv-ratio", type=float, default=1.0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.cal_nsamples, args.cal_max_seqlen, args.cal_batch_size) <= 0:
        parser.error("Calibration sizes must be positive")
    if args.qk_mqa_dim <= 0:
        parser.error("qk-mqa-dim must be positive")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use cpu or a single CUDA device")
    if not args.inspect and not args.save_path:
        parser.error("--save-path is required for conversion")
    if args.save_path:
        output = Path(args.save_path)
        if output.resolve() == Path(args.model_path).resolve() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
            parser.error("--save-path must be a new or empty directory, separate from the source")
    raw, _ = PretrainedConfig.get_config_dict(args.model_path, revision=args.revision, local_files_only=args.local_files_only)
    source = source_config_from_dict(raw)
    config = make_export_config(source, kv_lora_rank=args.kv_lora_rank, qk_mqa_dim=args.qk_mqa_dim,
                                q_lora_rank=args.q_lora_rank, source_model_type=raw["model_type"])
    validate_conversion(config, args.freqfold, args.balance_kv_ratio)
    report = {"source": {"path": args.model_path, "requested_revision": args.revision,
                          "resolved_revision": raw.get("_commit_hash"), "model_type": raw["model_type"]},
              "conversion": vars(args), "code": code_provenance(),
              "training_tokens": 0, "optimizer_steps": 0,
              "qk_norm_preserved": config.qk_norm_preserved,
              "implementation": "qwen3_norm_preserving_adaptation" if config.qk_norm_preserved else "upstream_linear_transmla",
              "cache_elements_per_layer_per_token": config.kv_lora_rank + config.qk_mqa_dim,
              "source_cache_elements_per_layer_per_token": 2 * source.num_key_value_heads * config.head_dim,
              "backend": "portable_transformers_eager_sdpa"}
    if args.inspect:
        print(json.dumps(report, indent=2))
        return report
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    batches, report["calibration"] = calibration_data(args, tokenizer)
    model, report["ignored_mtp_keys"] = load_source_model(args, raw)
    model = convert_model(model, batches, kv_lora_rank=args.kv_lora_rank, qk_mqa_dim=args.qk_mqa_dim,
                          freqfold=args.freqfold, q_lora_rank=args.q_lora_rank,
                          balance_kv_ratio=args.balance_kv_ratio, source_model_type=raw["model_type"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "safetensors")}
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved converted model, tokenizer, and conversion_report.json to {output}")
    return report


if __name__ == "__main__":
    main()
