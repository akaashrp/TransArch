"""Quality evaluations using the pinned Linearization protocol and HF backend."""
import argparse
import hashlib
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from unittest.mock import patch

from .common import RunStore, atomic_json, checkpoint_identity, digest, environment, file_hash, load_model, source_hash
from .hf_backend import HFGenerator, Sampling
from . import protocol


def summarize_math(items):
    correct = [c for p in items for c in p["correct"]]
    truncated = [c["truncated"] for p in items for c in p["outputs"]]
    counts = [c["gen_tokens"] for p in items for c in p["outputs"]]
    n = len(items[0]["correct"])
    result = {"accuracy": sum(correct) / len(correct), "pass@1": sum(correct) / len(correct),
              "n_problems": len(items), "n_samples": n,
              "truncated_frac": sum(truncated) / len(truncated),
              "mean_gen_tokens": sum(counts) / len(counts)}
    if n >= 8:
        result["pass@8"] = sum(1 - (math.comb(n - sum(p["correct"]), 8) / math.comb(n, 8)
                                   if n - sum(p["correct"]) >= 8 else 0) for p in items) / len(items)
    return result


def summarize_ruler(items):
    values = [p["score"] for p in items]
    mean = sum(values) / len(values)
    # Item-level SE handles fractional multiquery recall without pretending
    # each query in an item is an independent example.
    se = (sum((x - mean)**2 for x in values) / (len(values) * (len(values) - 1)))**0.5 if len(values) > 1 else None
    if all(len(p.get("expected", [None])) == 1 for p in items):
        z, n = 1.959963984540054, len(items)
        center = (mean + z*z/(2*n)) / (1 + z*z/n)
        half = z * (mean*(1-mean)/n + z*z/(4*n*n))**0.5 / (1 + z*z/n)
        interval, method = [max(0, center-half), min(1, center+half)], "Wilson over items"
    else:
        import numpy as np
        draws = np.random.default_rng(1234).choice(values, size=(2000, len(values)), replace=True).mean(1)
        interval, method = np.quantile(draws, [0.025, 0.975]).tolist(), "item bootstrap, 2000 draws, seed 1234"
    return {"accuracy": mean, "n_items": len(items), "item_standard_error": se,
            "accuracy_ci95": interval, "ci_method": method,
            "truncated_frac": sum(p["outputs"][0]["truncated"] for p in items) / len(items),
            "mean_gen_tokens": sum(p["outputs"][0]["gen_tokens"] for p in items) / len(items)}


def run_reasoning(args, model, tok, store):
    generator = HFGenerator(model, tok)
    if args.task.startswith("niah_"):
        from transformers import AutoTokenizer
        item_tok = AutoTokenizer.from_pretrained(args.item_tokenizer, local_files_only=True, trust_remote_code=True)
        rng = random.Random(f"{args.seed}:{args.task}:{args.length}")
        rows = [protocol.make_ruler_item(args.task, item_tok, args.length - 128 - 32, rng)
                for _ in range(args.limit)]
        sampling = Sampling(max_tokens=args.max_tokens)
    else:
        rows = [json.loads(line) for line in (Path(args.data) / f"{args.task}.jsonl").read_text().splitlines()]
        if args.offset < 0 or args.offset >= len(rows):
            raise ValueError("Math offset outside dataset")
        rows = rows[args.offset:args.offset + args.limit]
        sampling = Sampling(args.n, 0.6 if args.thinking else 0.7,
                            0.95 if args.thinking else 0.8, 20, args.max_tokens)
    items = []
    for index, row in enumerate(rows, args.offset):
        content = row["input"] if args.task.startswith("niah_") else row.get("problem", row.get("question")) + protocol.MATH_INSTRUCTION
        prompt = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=args.thinking)
        if args.task.startswith("niah_"):
            prompt += row["prefill"]
        key = str(index)
        item = store.get(key)
        if item is None:
            seed = int(digest([args.seed, args.task, args.thinking, index])[:8], 16)
            result = generator.generate(prompt, sampling, seed)
            item = {"idx": index, "prompt_sha256": digest(prompt),
                    "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "sampling": asdict(sampling), **result}
            if args.task.startswith("niah_"):
                item.update(expected=row["expected"], **protocol.score_ruler_item(row, result["outputs"][0]["text"]))
            else:
                gold = str(row["answer"]).rsplit("####", 1)[-1].strip()
                item.update(gold=gold, correct=[protocol.score_math(gold, c["text"]) for c in result["outputs"]])
            store.put(key, item)
        elif item["prompt_sha256"] != digest(prompt):
            raise ValueError("Cached item's prompt changed")
        items.append(item)
        print(f"{args.task}: {len(items)}/{len(rows)}", flush=True)
    summary = summarize_ruler(items) if args.task.startswith("niah_") else summarize_math(items)
    atomic_json(Path(args.out) / "results.json", {"status": "complete", "task": args.task,
                "summary": summary, "items": items})


@contextmanager
def offline_harness_data(data):
    """Use staged DatasetDicts, keeping upstream task configurations unchanged."""
    import datasets
    registry = json.loads((Path(data) / "harness_registry.json").read_text())
    def local_load(path, name=None, **kwargs):
        key = digest([path, name])
        if key not in registry:
            raise ValueError(f"Dataset not staged: {path}/{name}")
        ds = datasets.load_from_disk(str(Path(data) / registry[key]["directory"]))
        if "split" in kwargs and kwargs["split"] is not None:
            return ds[kwargs["split"]]
        return ds
    with patch.object(datasets, "load_dataset", local_load):
        yield


def run_likelihood(args, model, tok):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    result_path = Path(args.out) / "results.json"
    if result_path.exists():
        return
    gsm8k = args.task == "gsm8k"
    wrapper = HFLM(pretrained=model, tokenizer=tok, batch_size=1,
                   max_length=min(model.config.max_position_embeddings, 8192),
                   enable_thinking=False if gsm8k else None)
    with offline_harness_data(args.data):
        result = lm_eval.simple_evaluate(
            model=wrapper, tasks=[args.task], num_fewshot=5 if args.task in ("mmlu", "gsm8k") else None,
            gen_kwargs="max_gen_toks=1024,do_sample=False" if gsm8k else None,
            # Keep the original five demonstrations in one user message;
            # the native assistant prefix explicitly closes the think block.
            apply_chat_template=gsm8k, fewshot_as_multiturn=False,
            limit=args.limit if args.limit else None, batch_size=1,
            log_samples=True, bootstrap_iters=1000, random_seed=0,
            numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
    atomic_json(result_path, {"status": "complete", "task": args.task, "lm_eval": result})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--task", required=True, choices=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu", "gsm8k", "math500", "aime24", "aime25", "niah_single", "niah_multikey", "niah_multiquery"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--item-tokenizer")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.task == "gsm8k" and args.thinking:
        raise ValueError("GSM8K uses explicit no-thinking chat prompts")
    reasoning = args.task in protocol.MATH_TASKS or args.task.startswith("niah_")
    reasoning = reasoning and args.task != "gsm8k"
    if reasoning and args.limit <= 0:
        raise ValueError("Reasoning jobs require a positive shard size")
    if args.task.startswith("niah_") and (not args.item_tokenizer or args.thinking or args.offset):
        raise ValueError("NIAH requires a fixed item tokenizer, no thinking, and offset zero")
    manifest = {"arguments": vars(args), "checkpoint": checkpoint_identity(args.model),
                "data_manifest_sha256": file_hash(Path(args.data) / "manifest.json"),
                "code_sha256": source_hash(), "environment": environment(), "backend": "hf"}
    store = RunStore(args.out, manifest)
    model, tok = load_model(args.model, args.device, args.dtype)
    if reasoning:
        run_reasoning(args, model, tok, store)
    else:
        run_likelihood(args, model, tok)


if __name__ == "__main__":
    main()
