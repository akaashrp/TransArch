"""Full-weight GPU gates. Invoked by prepared jobs, never implicitly submitted."""
import argparse
import gc
import json
from pathlib import Path

import torch

from .common import atomic_json, checkpoint_identity, environment, load_model, source_hash


def probe_tokens(tokenizer, data, length=64):
    rows = [json.loads(s)["text"] for s in (Path(data) / "validation.jsonl").read_text().splitlines()]
    text = "\n\n".join(s for s in rows if s.strip())
    ids = tokenizer(text[:max(8192, length * 12)], add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] < length:
        raise ValueError("Not enough held-out probe text")
    return ids[:, :length]


@torch.inference_mode()
def check_model(model, ids, converted=False):
    ids = ids.to(model.device)
    model.config.mla_prefill_backend = "chunked"
    reference = model(ids, use_cache=False).logits.float()
    if not torch.isfinite(reference).all():
        raise ValueError("Nonfinite full-model logits")
    prefix = model(ids[:, :17], use_cache=True)
    suffix = model(ids[:, 17:], past_key_values=prefix.past_key_values, use_cache=True).logits.float()
    torch.testing.assert_close(suffix, reference[:, 17:], atol=0.08, rtol=0.04)
    mask = torch.cat((torch.zeros(1, 3, device=model.device, dtype=torch.long), torch.ones_like(ids)), 1)
    padded = torch.cat((torch.full_like(ids[:, :3], model.config.pad_token_id or 0), ids), 1)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    logits = model(padded, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, 3:].float()
    torch.testing.assert_close(logits, reference, atol=0.08, rtol=0.04)
    report = {"cached_max_abs_error": (suffix - reference[:, 17:]).abs().max().item(),
              "padded_max_abs_error": (logits - reference).abs().max().item()}
    if converted:
        for layer in prefix.past_key_values.layers:
            if layer.keys.shape[-1] != model.config.kv_lora_rank or layer.values.shape[-1] != model.config.qk_mqa_dim:
                raise ValueError("Cache is not stored in latent representation")
        model.config.mla_prefill_backend = "auto"
        fused = model(ids, use_cache=False).logits.float()
        torch.testing.assert_close(fused, reference, atol=0.08, rtol=0.04)
        report["prefill_max_abs_error"] = (fused - reference).abs().max().item()
        report["prefill_backends"] = sorted({l.self_attn.last_attention_backend for l in model.model.layers})
    return report, reference.cpu()


@torch.inference_mode()
def run_validation(model_path, data, out, kind, lengths=(4096, 8192, 16384, 32768)):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation requires CUDA; CPU tests cannot release experiment jobs")
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(model_path)
    raw = json.loads((Path(model_path) / "config.json").read_text())
    ids = probe_tokens(tok, data)
    report, reference = check_model(model, ids, converted=kind == "converted")
    if kind == "converted":
        before_save = torch.load(Path(model_path) / "validation_reference.pt", map_location="cpu", weights_only=True)
        if not torch.equal(ids, before_save["input_ids"]):
            raise ValueError("Held-out reload probe changed")
        torch.testing.assert_close(reference, before_save["logits"].float(), atol=0.08, rtol=0.04)
        report["save_reload_max_abs_error"] = (reference - before_save["logits"].float()).abs().max().item()
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
