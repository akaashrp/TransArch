"""Prepare, inspect, and explicitly submit a dependency-ordered Slurm campaign.

Preparation and dry-run never contact the scheduler. Only launch --execute
submits jobs. Every worker independently verifies its prerequisite GPU gates.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .common import atomic_json, digest, environment, file_hash, source_hash

PROJECT = Path(__file__).resolve().parents[2]
SOURCES = {
    "qwen3": {"repo": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c"},
    "mimo": {"repo": "XiaomiMiMo/MiMo-7B-RL-0530", "revision": "323400599af3903adc2a536d6340a23fee88d2e0"},
}
STAGES = ("source", "convert", "diagnostic", "full")


def build_jobs(root, sources, data, ranks):
    jobs = {stage: [] for stage in STAGES}
    rows = []
    for family, source in sources.items():
        gate = str(root / "gates" / f"{family}-source.json")
        jobs["source"].append({"family": family, "model": source["path"], "gate": gate})
        rows.append({"id": f"{family}-teacher", "family": family, "method": "teacher", "model": source["path"], "gate": gate})
        for rank in ranks:
            name = f"{family}-transmla-r{rank}"
            model = str(root / "models" / name)
            converted_gate = model + ".gpu-validation.json"
            jobs["convert"].append({"family": family, "source": source["path"], "source_gate": gate,
                                     "model": model, "rank": rank, "gate": converted_gate})
            rows.append({"id": name, "family": family, "method": "transmla", "model": model,
                         "gate": converted_gate, "rank": rank, "rope_dim": 64, "freqfold": 4})
    for row in rows:
        jobs["diagnostic"].append({"row": row})
        for task in ("piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu", "gsm8k"):
            jobs["full"].append({"row": row, "task": task, "limit": 0, "offset": 0, "thinking": False,
                                "n": 1, "max_tokens": 1024, "seed": 0})
        for thinking, cap in ((True, 32768), (False, 4096)):
            for offset in range(0, 500, 25):
                jobs["full"].append({"row": row, "task": "math500", "limit": 25, "offset": offset,
                                    "thinking": thinking, "n": 1, "max_tokens": cap, "seed": 0})
        for task in ("aime24", "aime25"):
            for offset in range(0, 30, 5):
                jobs["full"].append({"row": row, "task": task, "limit": 5, "offset": offset,
                                    "thinking": True, "n": 8, "max_tokens": 32768, "seed": 0})
        for task in ("niah_single", "niah_multikey", "niah_multiquery"):
            for length in (4096, 8192, 16384, 32768):
                for seed in range(5):
                    jobs["full"].append({"row": row, "task": task, "length": length, "limit": 100,
                                        "offset": 0, "thinking": False, "n": 1, "max_tokens": 128, "seed": seed})
    for stage, entries in jobs.items():
        for index, job in enumerate(entries):
            job["index"] = index
            job["output"] = str(root / "results" / stage / f"{index:04d}")
    return rows, jobs


def prepare(args):
    from huggingface_hub import snapshot_download
    root = Path(args.out).resolve()
    if (root / "submission.json").exists():
        raise ValueError("Cannot rewrite a campaign with a submission journal")
    sources = {family: {**entry, "path": snapshot_download(entry["repo"], revision=entry["revision"], local_files_only=True)}
               for family, entry in SOURCES.items()}
    data = Path(args.data).resolve()
    rows, jobs = build_jobs(root, sources, data, [512, 1024] if args.include_milder else [512])
    data_manifest = json.loads((data / "manifest.json").read_text())
    plan = {"schema_version": 1, "root": str(root), "project": str(PROJECT), "data": str(data),
            "data_manifest_sha256": file_hash(data / "manifest.json"),
            "sources": sources, "rows": rows, "jobs": jobs, "code_sha256": source_hash(),
            "environment": environment(), "item_tokenizer": sources["qwen3"]["path"],
            "resources": {"account": args.account, "partition": "GPU-shared", "qos": "gpu",
                          "gres": "gpu:h100-80:1", "cpus": 8, "memory": "128G", "concurrency": 2},
            "protocol": {"calibration": "128 sequences capped at 256 tokens, train split, seed 42; actual token count recorded",
                         "validation": "8 sequences capped at 256 tokens, validation split, seed 43",
                         "ruler": "no thinking, no answer prefill, 128 answer tokens for every row, seeds 0..4",
                         "harness": "lm_eval 0.4.12 defaults; MMLU 5-shot, GSM8K default 5-shot, fixed fewshot seed 1234",
                         "generation": "HF; per-problem seed derived from task, mode, index, and seed; native EOS",
                         "comparison": "Teacher controls recomputed in HF; legacy vLLM results require protocol audit"},
            "external_comparators": {"swa": "Anthony's code/artifacts, windows and layer masks required",
                                     "gdn_opd": "Selected paper checkpoint and raw evaluation artifacts required"},
            "prepared_only": True}
    existing = root / "plan.json"
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise ValueError("Prepared plan changed; choose a fresh output directory")
    atomic_json(existing, plan)
    for stage, entries in jobs.items():
        script = root / f"{stage}.sbatch"
        wall = {"source": "02:00:00", "convert": "08:00:00", "diagnostic": "02:00:00", "full": "24:00:00"}[stage]
        script.write_text("#!/usr/bin/env bash\n"
                          f"#SBATCH --job-name=transmla-{stage}\n#SBATCH --account={args.account}\n"
                          "#SBATCH --partition=GPU-shared\n#SBATCH --qos=gpu\n#SBATCH --gres=gpu:h100-80:1\n"
                          f"#SBATCH --cpus-per-task=8\n#SBATCH --mem=128G\n#SBATCH --time={wall}\n"
                          f"#SBATCH --array=0-{len(entries)-1}%2\n#SBATCH --output={root}/logs/{stage}-%A_%a.out\n"
                          "set -euo pipefail\n"
                          f"source {shlex.quote(str(PROJECT / 'scripts/eval_env.sh'))}\n"
                          f'"$TRANSMLA_PYTHON" -m transmla.experiments.campaign worker --plan {shlex.quote(str(existing))}'
                          f' --stage {stage} --index "$SLURM_ARRAY_TASK_ID"\n')
    (root / "logs").mkdir(exist_ok=True)
    (root / "submit.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                                    f"source {shlex.quote(str(PROJECT / 'scripts/eval_env.sh'))}\n"
                                    f'"$TRANSMLA_PYTHON" -m transmla.experiments.campaign launch --plan {shlex.quote(str(existing))} "$@"\n')
    atomic_json(root / "preflight.json", preflight(plan))
    print(json.dumps({"plan": str(existing), "jobs": {s: len(j) for s,j in jobs.items()}, "submitted": False}, indent=2))


def preflight(plan):
    if plan["code_sha256"] != source_hash() or plan["environment"] != environment():
        raise ValueError("Code/environment changed since preparation; regenerate in a new directory")
    data = Path(plan["data"])
    if file_hash(data / "manifest.json") != plan["data_manifest_sha256"]:
        raise ValueError("Data manifest changed")
    for name, expected in json.loads((data / "manifest.json").read_text())["files"].items():
        if file_hash(data / name) != expected:
            raise ValueError(f"Staged data changed: {name}")
    from safetensors import safe_open
    sources = {}
    for family, source in plan["sources"].items():
        directory = Path(source["path"])
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        for name in shards:
            with safe_open(directory / name, framework="pt", device="cpu") as f:
                actual = set(f.keys())
                expected = {k for k, v in index["weight_map"].items() if v == name}
                if actual != expected:
                    raise ValueError(f"Shard inventory mismatch: {directory / name}")
        sources[family] = {"revision": source["revision"], "shards": len(shards), "tensors": len(index["weight_map"])}
    return {"status": "passed_cpu_preflight", "sources": sources, "gpu_validation": "not_run",
            "scheduler_submission": "not_performed"}


def evaluation_args(plan, job, out):
    argv = ["--model", job["row"]["model"], "--data", plan["data"], "--out", out,
            "--task", job["task"], "--item-tokenizer", plan["item_tokenizer"]]
    for key in ("limit", "offset", "n", "max_tokens", "seed", "length"):
        if key in job:
            argv += ["--" + key.replace("_", "-"), str(job[key])]
    if job.get("thinking"):
        argv += ["--thinking"]
    return argv


def worker(plan, stage, index):
    from .validate import require_gate, run_validation
    if plan["code_sha256"] != source_hash() or plan["environment"] != environment():
        raise ValueError("Worker code/environment differs from prepared plan")
    if file_hash(Path(plan["data"]) / "manifest.json") != plan["data_manifest_sha256"]:
        raise ValueError("Worker data manifest differs from prepared plan")
    if index < 0 or index >= len(plan["jobs"][stage]):
        raise ValueError("Worker array index outside the prepared stage")
    job = plan["jobs"][stage][index]
    if stage == "source":
        run_validation(job["model"], plan["data"], job["gate"], "source")
    elif stage == "convert":
        from .convert import main
        main(["--source", job["source"], "--source-gate", job["source_gate"], "--out", job["model"],
              "--data", plan["data"], "--rank", str(job["rank"])])
    else:
        row = job["row"]
        require_gate(row["gate"], row["model"])
        diagnostic_gate = Path(plan["root"]) / "gates" / f"{row['id']}-diagnostic.json"
        if stage == "diagnostic":
            for task in ("piqa", "math500", "niah_multikey"):
                probe = {"row": row, "task": task, "limit": 2, "max_tokens": 128,
                         "thinking": False, "n": 1, "offset": 0, "seed": 0, "length": 4096}
                argv = evaluation_args(plan, probe, str(Path(job["output"]) / task))
                subprocess.run([sys.executable, "-m", "transmla.experiments.evaluate", *argv], check=True)
            atomic_json(diagnostic_gate, {"status": "passed", "plan_sha256": digest(plan), "row": row})
        else:
            gate = json.loads(diagnostic_gate.read_text())
            if gate != {"status": "passed", "plan_sha256": digest(plan), "row": row}:
                raise ValueError("Small diagnostic evaluation has not passed for this plan")
            subprocess.run([sys.executable, "-m", "transmla.experiments.evaluate",
                            *evaluation_args(plan, job, job["output"])], check=True)


def launch(plan, execute=False):
    preflight(plan)
    root = Path(plan["root"])
    journal = root / "submission.json"
    if execute and journal.exists():
        raise ValueError("Submission journal already exists; use the recorded job IDs and explicit shard resubmission")
    previous, submitted = None, {}
    for stage in STAGES:
        command = ["sbatch", "--parsable"]
        if previous:
            command += [f"--dependency=afterok:{previous}"]
        command += [str(root / f"{stage}.sbatch")]
        print(shlex.join(command), flush=True)
        if execute:
            output = subprocess.check_output(command, text=True).strip()
            previous = output.split(";", 1)[0]
            if not previous.isdigit():
                raise RuntimeError(f"Unrecognized sbatch response: {output}")
            submitted[stage] = previous
            atomic_json(journal, {"plan_sha256": digest(plan), "jobs": submitted})
        else:
            previous = f"<{stage}_job_id>"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--out", required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--account", default="cis260115p")
    q.add_argument("--include-milder", action="store_true")
    for name in ("preflight", "worker", "launch"):
        q = sub.add_parser(name)
        q.add_argument("--plan", required=True)
        if name == "worker":
            q.add_argument("--stage", choices=STAGES, required=True)
            q.add_argument("--index", type=int, required=True)
        if name == "launch":
            q.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    else:
        plan = json.loads(Path(args.plan).read_text())
        if args.command == "preflight":
            print(json.dumps(preflight(plan), indent=2))
        elif args.command == "worker":
            worker(plan, args.stage, args.index)
        else:
            launch(plan, args.execute)


if __name__ == "__main__":
    main()
