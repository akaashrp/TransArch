"""Full-weight GPU gates. Invoked by prepared jobs, never implicitly submitted."""
import argparse
from contextlib import ExitStack, contextmanager
from functools import partial
import gc
import json
from pathlib import Path

import torch

from .common import atomic_json, checkpoint_identity, environment, load_model, source_hash

FP32_ATOL, FP32_RTOL = 0.002, 0.0001


@contextmanager
def fp32_reference_backend():
    """Keep high-precision structural checks independent of CUDA backend selection."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def probe_tokens(tokenizer, data, length=64):
    rows = [json.loads(s)["text"] for s in (Path(data) / "validation.jsonl").read_text().splitlines()]
    text = "\n\n".join(s for s in rows if s.strip())
    ids = tokenizer(text[:max(8192, length * 12)], add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] < length:
        raise ValueError("Not enough held-out probe text")
    return ids[:, :length]


@torch.inference_mode()
def export_reference(model, ids):
    """After saving BF16 weights, promote the disposable model for its FP32 oracle."""
    previous = model.config.mla_prefill_backend
    try:
        model.config.mla_prefill_backend = "chunked"
        model.float()
        with fp32_reference_backend():
            logits = model(ids.to(model.device), use_cache=False).logits.cpu()
    finally:
        model.config.mla_prefill_backend = previous
    return {"input_ids": ids.cpu(), "logits": logits, "prefill_backend": "chunked",
            "dtype": "float32", "tf32": False}


def logit_diagnostics(actual, expected):
    """Record BF16 shape/backend drift without pretending it is elementwise parity."""
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Nonfinite runtime logits")
    delta = (actual.float() - expected.float()).abs()
    a, b = actual.float().log_softmax(-1), expected.float().log_softmax(-1)
    disagreements = int((actual.argmax(-1) != expected.argmax(-1)).sum())
    positions = actual.numel() // actual.shape[-1]
    return {"max_abs_error": delta.max().item(), "rms_error": delta.square().mean().sqrt().item(),
            "max_kl": (a.exp() * (a - b)).sum(-1).clamp_min(0).max().item(),
            "top1_disagreements": disagreements, "positions": positions}


@torch.inference_mode()
def check_model(model, ids, converted=False, *, atol=FP32_ATOL, rtol=FP32_RTOL,
                check_prefill=True, enforce_parity=True):
    ids = ids.to(model.device)
    model.config.mla_prefill_backend = "chunked"
    reference = model(ids, use_cache=False).logits.float()
    if not torch.isfinite(reference).all():
        raise ValueError("Nonfinite full-model logits")
    prefix = model(ids[:, :17], use_cache=True)
    suffix = model(ids[:, 17:], past_key_values=prefix.past_key_values, use_cache=True).logits.float()
    cached = logit_diagnostics(suffix, reference[:, 17:])
    if enforce_parity:
        torch.testing.assert_close(suffix, reference[:, 17:], atol=atol, rtol=rtol)
    mask = torch.cat((torch.zeros(1, 3, device=model.device, dtype=torch.long), torch.ones_like(ids)), 1)
    padded = torch.cat((torch.full_like(ids[:, :3], model.config.pad_token_id or 0), ids), 1)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    logits = model(padded, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, 3:].float()
    padded = logit_diagnostics(logits, reference)
    if enforce_parity:
        torch.testing.assert_close(logits, reference, atol=atol, rtol=rtol)
    report = {"cached_max_abs_error": cached["max_abs_error"],
              "padded_max_abs_error": padded["max_abs_error"], "cached": cached, "padded": padded}
    if converted:
        for layer in prefix.past_key_values.layers:
            if layer.keys.shape[-1] != model.config.kv_lora_rank or layer.values.shape[-1] != model.config.qk_mqa_dim:
                raise ValueError("Cache is not stored in latent representation")
        if check_prefill:
            model.config.mla_prefill_backend = "auto"
            fused = model(ids, use_cache=False).logits.float()
            report["prefill"] = logit_diagnostics(fused, reference)
            if enforce_parity:
                torch.testing.assert_close(fused, reference, atol=atol, rtol=rtol)
            report["prefill_max_abs_error"] = report["prefill"]["max_abs_error"]
            report["prefill_backends"] = sorted({l.self_attn.last_attention_backend for l in model.model.layers})
    return report, reference.cpu()


@torch.inference_mode()
def check_fp32_model(model, ids, converted=False):
    if model.dtype != torch.float32:
        raise ValueError("Structural reference gate requires an FP32 model")
    with fp32_reference_backend():
        report, reference = check_model(model, ids, converted, check_prefill=False)
        if converted:
            # Exercise the expanded prefill algebra in FP32 even though Flash
            # itself only accepts FP16/BF16. Patch only this short oracle pass.
            from unittest.mock import patch
            def expanded_math(attention, q, qr, latent, kr, ku, vu, mask, output_attentions):
                if mask is not None or output_attentions or q.shape[-2] != latent.shape[-2]:
                    raise ValueError("Expanded FP32 oracle requires an unmasked full prefill")
                query, key, value = attention._expanded_qkv(q, qr, latent, kr, ku, vu)
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True, scale=attention.scaling)
                return output[..., :attention.head_dim]
            with ExitStack() as stack:
                for layer in model.model.layers:
                    attention = layer.self_attn
                    stack.enter_context(patch.object(attention, "_flash_prefill", partial(expanded_math, attention)))
                expanded = model(ids.to(model.device), use_cache=False).logits.float().cpu()
            torch.testing.assert_close(expanded, reference, atol=FP32_ATOL, rtol=FP32_RTOL)
            report["expanded_math_max_abs_error"] = (expanded - reference).abs().max().item()
    report.update(dtype="float32", attention_backend="math/chunked", tf32=False, atol=FP32_ATOL, rtol=FP32_RTOL)
    return report, reference


@torch.inference_mode()
def run_validation(model_path, data, out, kind, lengths=(4096, 8192, 16384, 32768)):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation requires CUDA; CPU tests cannot release experiment jobs")
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(model_path)
    raw = json.loads((Path(model_path) / "config.json").read_text())
    ids = probe_tokens(tok, data)
    report, reference = check_model(model, ids, converted=kind == "converted", enforce_parity=False)
    report["bf16_comparisons"] = "finite runtime checks and recorded numerical drift; structural parity checked in FP32"
    before_save = None
    if kind == "converted":
        before_save = torch.load(Path(model_path) / "validation_reference.pt", map_location="cpu", weights_only=True)
        if (before_save.get("prefill_backend") != "chunked" or before_save.get("dtype") != "float32"
                or before_save.get("tf32") is not False):
            raise ValueError("Export reference must use FP32 chunked attention with TF32 disabled")
        if not torch.equal(ids, before_save["input_ids"]):
            raise ValueError("Held-out reload probe changed")
    if kind == "source" and raw["model_type"] == "mimo":
        del model
        gc.collect()
        torch.cuda.empty_cache()
        native, _ = load_model(model_path, native_mimo=True)
        native_logits = native(ids.to(native.device), use_cache=False).logits.float().cpu()
        torch.testing.assert_close(native_logits, reference, atol=0.005, rtol=0.005)
        report["native_mimo_max_abs_error"] = (native_logits - reference).abs().max().item()
        del native
        gc.collect()
        torch.cuda.empty_cache()
        model, tok = load_model(model_path)
    if kind == "converted":
        long_results = []
        for length in lengths:
            tokens = probe_tokens(tok, data, length).to(model.device)
            model.config.mla_prefill_backend = "auto"
            result = model(tokens, use_cache=True, logits_to_keep=1)
            if not torch.isfinite(result.logits).all():
                raise ValueError("Nonfinite long-context logits")
            backends = sorted({l.self_attn.last_attention_backend for l in model.model.layers})
            # H100 campaign: fail early if fused prefill unexpectedly falls back.
            if backends != ["flash_expanded_prefill"]:
                raise RuntimeError(f"Expected fused CUDA prefill at {length}: {backends}")
            elements = sum(l.keys.numel() + l.values.numel() for l in result.past_key_values.layers)
            expected = length * model.config.num_hidden_layers * (model.config.kv_lora_rank + model.config.qk_mqa_dim)
            if elements != expected:
                raise ValueError("Unexpected logical cache size")
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            decoded = model(token, past_key_values=result.past_key_values, use_cache=True, logits_to_keep=1)
            if not torch.isfinite(decoded.logits).all():
                raise ValueError("Nonfinite long-context decode")
            long_results.append({"length": length, "backends": backends, "cache_elements": elements})
            del result, decoded, tokens
        report["long_context"] = long_results
    # Exercise the actual HF generation path with the saved tokenizer/config.
    from .hf_backend import HFGenerator, Sampling
    prompt = tok.apply_chat_template([{"role": "user", "content": "What is 2 + 2?"}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    report["generation_smoke"] = HFGenerator(model, tok).generate(prompt, Sampling(max_tokens=16), 42)
    # Promote only after BF16 generation/long-context checks; never downcast
    # this model back, which would also round the FP32 rotary-frequency buffers.
    precise, fp32_reference = check_fp32_model(model.float(), ids, converted=kind == "converted")
    report["fp32_structural"] = precise
    if before_save is not None:
        torch.testing.assert_close(fp32_reference, before_save["logits"].float(), atol=FP32_ATOL, rtol=FP32_RTOL)
        report["save_reload_max_abs_error"] = (fp32_reference - before_save["logits"].float()).abs().max().item()
        report["save_reload_dtype"] = "float32"
    report["validation_policy"] = "fp32-structural-bf16-runtime-v1"
    report.update(status="passed", kind=kind, checkpoint=checkpoint_identity(model_path),
                  code_sha256=source_hash(), environment=environment(),
                  gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(out, report)
    return report


def require_gate(path, model_path):
    report = json.loads(Path(path).read_text())
    if (report.get("status") != "passed" or report.get("checkpoint") != checkpoint_identity(model_path)
            or report.get("code_sha256") != source_hash() or report.get("environment") != environment()):
        raise ValueError(f"Missing, stale, or failed GPU validation: {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", required=True, choices=["source", "converted"])
    args = p.parse_args(argv)
    run_validation(args.model, args.data, args.out, args.kind)


if __name__ == "__main__":
    main()
