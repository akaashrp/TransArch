"""Pinned calibration plus held-out stage diagnostics and atomic export."""
from copy import deepcopy
import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from ..convert_pretrained import calibration_data, code_provenance, convert_model
from ..utils import evaluate_ppl
from .common import atomic_json, checkpoint_identity, environment, file_hash, load_model
from .validate import probe_tokens, require_gate, run_validation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--source-gate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--rank", required=True, type=int)
    args = p.parse_args(argv)
    require_gate(args.source_gate, args.source)
    output = Path(args.out)
    requested = {"source": checkpoint_identity(args.source), "rank": args.rank,
                 "data": str(Path(args.data).resolve()), "data_sha256": file_hash(Path(args.data) / "manifest.json"),
                 "code": code_provenance(), "environment": environment()}
    if output.exists():
        report = json.loads((output / "conversion_report.json").read_text())
        if report["request"] != requested:
            raise ValueError("Existing conversion has a different identity")
    else:
        work = output.with_name(output.name + ".incomplete")
        if work.exists():
            raise ValueError(f"Incomplete export retained at {work}; inspect/move it before retrying")
        model, tok = load_model(args.source)
        settings = SimpleNamespace(calibration_file=str(Path(args.data) / "calibration.jsonl"),
                                   cal_max_seqlen=256, cal_batch_size=4, cal_nsamples=128, seed=42)
        batches, calibration = calibration_data(settings, tok)
        if calibration["samples"] != 128:
            raise ValueError("Calibration did not produce the configured 128 samples")
        heldout = deepcopy(settings)
        heldout.calibration_file = str(Path(args.data) / "validation.jsonl")
        heldout.cal_nsamples, heldout.seed = 8, 43
        probes, validation = calibration_data(heldout, tok)
        diagnostics = {}
        def callback(stage, current):
            value = evaluate_ppl(current, tok.pad_token_id, probes)
            if not 0 < value < float("inf"):
                raise ValueError(f"Nonfinite held-out perplexity at {stage}")
            diagnostics[stage] = value
        family = json.loads((Path(args.source) / "config.json").read_text())["model_type"]
        converted = convert_model(model, batches, kv_lora_rank=args.rank, qk_mqa_dim=64,
                                  freqfold=4, balance_kv_ratio=1.0, source_model_type=family,
                                  stage_callback=callback)
        converted.save_pretrained(work, safe_serialization=True, max_shard_size="4GB")
        tok.save_pretrained(work)
        with torch.inference_mode():
            ids = probe_tokens(tok, args.data)
            logits = converted(ids.to(converted.device), use_cache=False).logits.cpu()
        torch.save({"input_ids": ids, "logits": logits}, work / "validation_reference.pt")
        atomic_json(work / "conversion_report.json", {
            "request": requested, "source_model_type": family, "calibration": calibration,
            "validation": validation, "heldout_perplexity": diagnostics,
            "kv_lora_rank": args.rank, "qk_mqa_dim": 64, "freqfold": 4,
            "qk_norm_preserved": family == "qwen3", "q_lora_rank": None,
            "optimizer_steps": 0, "training_tokens": 0,
            "logical_cache_fraction": (args.rank + 64) / (2 * model.config.num_key_value_heads * model.config.head_dim)})
        os.rename(work, output)
        del converted, model, batches, probes
        gc.collect()
        torch.cuda.empty_cache()
    # Reload from the published local artifact and exercise full-size inference.
    run_validation(output, args.data, output.parent / f"{output.name}.gpu-validation.json", "converted")


if __name__ == "__main__":
    main()
