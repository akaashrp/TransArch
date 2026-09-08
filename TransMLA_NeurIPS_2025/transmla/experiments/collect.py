"""Collect completed shards without treating missing/failed jobs as zero."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .common import atomic_json, digest
from .campaign import evaluation_args
from .evaluate import parser, summarize_math, summarize_ruler


def collect(plan):
    groups = defaultdict(list)
    missing = []
    checkpoints = {}
    for job in plan["jobs"]["full"]:
        key = (job["row"]["id"], job["task"], job["thinking"], job.get("length"), job["max_tokens"])
        result = Path(job["output"]) / "results.json"
        if not result.exists():
            missing.append(job["index"])
            groups[key].append((job, None))
            continue
        report = json.loads(result.read_text())
        manifest = json.loads((result.parent / "manifest.json").read_text())
        arguments = manifest["arguments"]
        expected = vars(parser().parse_args(evaluation_args(plan, job, job["output"])))
        if (report["status"] != "complete" or arguments != expected or manifest.get("backend") != "hf"
                or manifest["code_sha256"] != plan["code_sha256"]
                or manifest["environment"] != plan["environment"]
                or manifest["data_manifest_sha256"] != plan["data_manifest_sha256"]):
            raise ValueError(f"Result identity mismatch: {result}")
        identity = manifest.get("checkpoint")
        if not isinstance(identity, dict) or identity.get("path") != str(Path(job["row"]["model"]).resolve()):
            raise ValueError(f"Checkpoint identity mismatch: {result}")
        if checkpoints.setdefault(job["row"]["id"], identity) != identity:
            raise ValueError(f"Mixed checkpoints in one model row: {result}")
        if "items" in report:
            items = report["items"]
            if len(items) != job["limit"] or any(len(item["outputs"]) != job["n"] for item in items):
                raise ValueError(f"Incomplete sample coverage in completed result: {result}")
            if {i["idx"] for i in items} != set(range(job["offset"], job["offset"] + job["limit"])):
                raise ValueError(f"Wrong problem indices: {result}")
        groups[key].append((job, report))
    rows = []
    for key, entries in sorted(groups.items()):
        row_id, task, thinking, length, cap = key
        completed = [r for _, r in entries if r is not None]
        row = {"row": row_id, "task": task, "thinking": thinking, "length": length, "max_tokens": cap,
               "completed_shards": len(completed), "expected_shards": len(entries),
               "status": "complete" if len(completed) == len(entries) else "partial"}
        if completed:
            if "lm_eval" in completed[0]:
                metrics = completed[0]["lm_eval"].get("results", {})
                groups_metrics = completed[0]["lm_eval"].get("groups", {})
                m = groups_metrics.get(task, metrics.get(task, {}))
                row["metrics"] = m
                row["accuracy"] = m.get("acc_norm,none", m.get("acc,none", m.get("exact_match,strict-match")))
            else:
                items = [item for r in completed for item in r["items"]]
                if not task.startswith("niah_") and len({i["idx"] for i in items}) != len(items):
                    raise ValueError(f"Duplicate math problems: {key}")
                if task.startswith("niah_") and len({i["input_sha256"] for i in items}) != len(items):
                    raise ValueError(f"Duplicate retrieval items: {key}")
                row.update(summarize_ruler(items) if task.startswith("niah_") else summarize_math(items))
        rows.append(row)
    lookup = {(r["row"], r["task"], r["thinking"], r["length"], r["max_tokens"]): r for r in rows}
    for row in rows:
        teacher_id = row["row"].split("-", 1)[0] + "-teacher"
        teacher = lookup.get((teacher_id, row["task"], row["thinking"], row["length"], row["max_tokens"]))
        if teacher and row["status"] == teacher["status"] == "complete" and row.get("accuracy") is not None and teacher.get("accuracy") is not None:
            row["delta_from_teacher"] = row["accuracy"] - teacher["accuracy"]
    return {"plan_sha256": digest(plan), "status": "complete" if not missing else "partial",
            "missing_job_indices": missing, "rows": rows, "external_comparators": plan["external_comparators"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    report = collect(plan)
    root = Path(args.out)
    atomic_json(root / "results.json", report)
    lines = ["# TransMLA quality comparison", "", f"Status: **{report['status']}**. Missing cells are not scores.", "",
             "| Model | Task | Mode | Context | Answer cap | Score (%) | Teacher delta (pp) | Shards |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        score = f"{100 * row['accuracy']:.2f}" if row.get("accuracy") is not None else "—"
        delta = f"{100 * row['delta_from_teacher']:+.2f}" if "delta_from_teacher" in row else "—"
        lines.append(f"| {row['row']} | {row['task']} | {'think' if row['thinking'] else 'no-think'} | "
                     f"{row['length'] or '—'} | {row['max_tokens']} | {score} | {delta} | "
                     f"{row['completed_shards']}/{row['expected_shards']} |")
    lines += ["", "SWA and the selected GDN+OPD row require the external artifacts listed in the plan.",
              "Do not merge historical scores until prompts, seeds, revisions, answer budgets and backend controls have been audited."]
    (root / "TABLES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
