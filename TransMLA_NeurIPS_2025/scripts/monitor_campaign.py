#!/usr/bin/env python3
"""Poll one submitted campaign and queue deduplicated alerts to its Codex session.

Standard library only. Run periodically with --config; --preview reads Slurm
without changing monitor state or sending a message. Never submits GPU jobs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess


TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED", "SPECIAL_EXIT",
}
BLOCKED = {"DependencyNeverSatisfied", "JobHeldAdmin", "JobHeldUser", "BadConstraints",
           "InvalidAccount", "InvalidQOS"}
STAGES = ("source", "convert", "diagnostic", "full")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.writing")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} exited {result.returncode}: {result.stderr.strip()[:1500]}")
    return result.stdout


def task_ids(value):
    """Expand Slurm's compressed array names, ignoring parent/step records."""
    match = re.fullmatch(r"(\d+)_(\d+|\[[\d,:%-]+\])", value)
    if not match:
        return []
    base, expression = match.groups()
    expression = expression.strip("[]").split("%", 1)[0]
    result = []
    for part in expression.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", part)
        if not match:
            raise ValueError(f"Unsupported Slurm array: {value}")
        start, end, step = match.groups()
        result.extend(f"{base}_{i}" for i in range(int(start), int(end or start) + 1, int(step or 1)))
    return result


def parse_records(accounting, queue):
    records = {}
    for line in accounting.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3:
            continue
        job_id, state, exit_code = fields[:3]
        for task in task_ids(job_id) or ([job_id] if job_id.isdigit() else []):
            records[task] = {"state": state.split()[0].rstrip("+"), "exit_code": exit_code,
                             "reason": "", "dependency": ""}
    # The live queue wins over accounting, which can lag during requeue/start.
    for line in queue.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        job_id, state, reason, dependency = fields[:4]
        for task in task_ids(job_id) or ([job_id] if job_id.isdigit() else []):
            records[task] = {"state": state, "exit_code": "", "reason": reason.strip("()"),
                             "dependency": dependency}
    return records


def summarize(jobs, counts, records):
    summaries, events = {}, {}
    all_terminal = True
    for stage in STAGES:
        base = jobs[stage]
        expected = [f"{base}_{i}" for i in range(counts[stage])]
        members = {task: records.get(task, {"state": "UNKNOWN"}) for task in expected}
        states = Counter(record["state"] for record in members.values())
        terminal = all(r["state"] in TERMINAL for r in members.values())
        all_terminal &= terminal
        summaries[stage] = {"job_id": base, "expected_tasks": counts[stage],
                            "states": dict(states), "all_terminal": terminal}
        prefix = f"{stage}:{base}"
        if states.get("RUNNING"):
            events[f"{prefix}:started"] = f"{stage} array {base} started"
        for task, record in members.items():
            state, reason = record["state"], record.get("reason", "")
            failed = state in TERMINAL and state != "COMPLETED"
            bad_exit = state == "COMPLETED" and record.get("exit_code", "0:0") not in ("", "0:0")
            if failed or bad_exit:
                events[f"{prefix}:{task}:{state}:{record.get('exit_code', '')}"] = (
                    f"{stage} {task}: {state}, exit {record.get('exit_code', '?')}")
            if reason in BLOCKED:
                # One blocked alert per array/reason, even for hundreds of tasks.
                events[f"{prefix}:blocked:{reason}"] = f"{stage} array {base}: {reason}"
        if terminal:
            label = "completed successfully" if states == {"COMPLETED": counts[stage]} and all(
                r.get("exit_code", "0:0") in ("", "0:0") for r in members.values()
            ) else "finished with failures/cancellations"
            events[f"{prefix}:terminal:{label}"] = f"{stage} array {base} {label} ({counts[stage]} tasks)"
    if all_terminal:
        events["campaign:terminal"] = "All submitted campaign tasks are terminal; inspect results and failures."
    return summaries, events, all_terminal


def send(config, message):
    output = command([config["codex"], "queue", "--thread", config["thread_id"], "--message", message])
    if "Queued message " not in output or config["thread_id"] not in output:
        raise RuntimeError(f"Unrecognized Codex delivery acknowledgement: {output[:1500]}")
    return output.strip()


def auxiliary_events(auxiliary, records):
    summaries, events = {}, {}
    for job in auxiliary:
        job_id = job["job_id"]
        record = records.get(job_id, {"state": "UNKNOWN"})
        summaries[job_id] = {"label": job["label"], **record}
        state = record["state"]
        reason = record.get("reason", "")
        if state == "RUNNING" or state in TERMINAL or reason in BLOCKED:
            key = f"auxiliary:{job_id}:{state}:{reason}:{record.get('exit_code', '')}"
            events[key] = f"{job['label']} {job_id}: {state}" + (f" ({reason})" if reason else "")
    return summaries, events


def pool_progress(override, plan_hash):
    directory = Path(override['pool'])
    manifest = json.loads((directory / 'pool.json').read_text())
    if manifest['plan_sha256'] != plan_hash or manifest['workers'] != override['task_count']:
        raise ValueError('Consolidated pool does not match this campaign')
    completed, failed = 0, []
    for index in manifest['indices']:
        path = directory / 'states' / f'{index:04d}.json'
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if state['status'] == 'complete':
            completed += 1
        elif state['status'] == 'failed':
            failed.append(index)
    return {'total_chunks': manifest['total_chunks'],
            'completed_chunks': manifest['completed_before_pool'] + completed,
            'unfinished_chunks': len(manifest['indices']) - completed,
            'failed_chunks': failed, 'pool': str(directory)}


def poll(config, directory, preview=False, baseline=False):
    bundle = Path(config["bundle"])
    journal = json.loads((bundle / "submission.json").read_text())
    jobs = dict(journal["jobs"])
    plan = json.loads((bundle / "plan.json").read_bytes())
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()
    if plan_hash != journal["plan_sha256"]:
        raise ValueError("Submitted plan hash changed; monitor cannot infer expected array sizes")
    counts = {k: len(v) for k, v in plan["jobs"].items()}
    overrides = config.get('stage_overrides', {})
    for stage, override in overrides.items():
        if stage != 'full' or override['task_count'] < 1 or not override['job_id'].isdigit():
            raise ValueError('Invalid execution stage override')
        jobs[stage], counts[stage] = override['job_id'], override['task_count']
    identity_source = {**journal, 'stage_overrides': overrides} if overrides else journal
    identity = hashlib.sha256(json.dumps(identity_source, sort_keys=True).encode()).hexdigest()
    state_path = directory / "state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get("identity") != identity:
        previous = {}
    auxiliary = config.get("auxiliary_jobs", [])
    if previous.get("all_terminal") and previous.get("auxiliary_jobs", []) == auxiliary and not preview:
        return  # A replaced submission journal automatically rearms the monitor.
    ids = [jobs[s] for s in STAGES] + [job["job_id"] for job in auxiliary]
    job_list = ",".join(ids)
    accounting = command([config["sacct"], "-X", "-j", job_list, "--starttime", config["accounting_start"],
                          "--format=JobID%128,State%64,ExitCode", "-P", "-n"])
    # Slurm returns an error for --jobs when all requested IDs leave the queue.
    # A user-filtered query succeeds when empty; --array avoids range truncation.
    queue = command([config["squeue"], "--user", config.get("user", getpass.getuser()),
                     "--array", "--noheader", "--format=%i|%T|%R|%E"])
    records = {task: record for task, record in parse_records(accounting, queue).items()
               if task.split("_", 1)[0] in ids}
    summaries, events, terminal = summarize(jobs, counts, records)
    progress = pool_progress(overrides['full'], plan_hash) if overrides else None
    if progress:
        for index in progress['failed_chunks']:
            events[f'pool:{jobs["full"]}:chunk:{index}:failed'] = f'Evaluation chunk {index} failed in consolidated pool; inspect {progress["pool"]}/logs.'
        if terminal and progress['unfinished_chunks']:
            events[f'pool:{jobs["full"]}:incomplete'] = f'GPU workers finished with {progress["unfinished_chunks"]} evaluation chunks unfinished; campaign results are incomplete.'
    extra_summaries, extra_events = auxiliary_events(auxiliary, records)
    events.update(extra_events)
    terminal = terminal and all(r["state"] in TERMINAL for r in extra_summaries.values())
    missing = [s for s, summary in summaries.items() if summary["states"].get("UNKNOWN")]
    unknown_polls = previous.get("unknown_polls", 0) + 1 if missing else 0
    if unknown_polls >= 3:
        events["campaign:unknown"] = "Slurm records missing for three checks: " + ", ".join(missing)
    seen = set(previous.get("delivered_events", []))
    fresh = {key: text for key, text in events.items() if key not in seen}
    status = {"checked_at": now(), "host": socket.gethostname(), "thread_id": config["thread_id"],
              "bundle": str(bundle), "identity": identity, "jobs": jobs, "stages": summaries,
              "records": records, "unknown_polls": unknown_polls, "all_terminal": terminal,
              "delivered_events": sorted(seen), "poll_interval_seconds": config["interval_seconds"],
              "auxiliary_jobs": auxiliary, "auxiliary_states": extra_summaries}
    if progress:
        status['evaluation_progress'] = progress
    if preview:
        print(json.dumps({"stages": summaries, "auxiliary_states": extra_summaries, 'evaluation_progress': progress,
                          "new_events": fresh, "all_terminal": terminal}, indent=2))
        return
    if baseline:
        # Setup has already delivered/reported these events; subsequent changes alert.
        status["delivered_events"] = sorted(seen | fresh.keys())
        status["baseline_at"] = now()
        atomic_json(state_path, status)
        atomic_json(directory / "status.json", status)
        return
    # Persist the snapshot before trying delivery; unsuccessful delivery is retried.
    atomic_json(directory / "status.json", status)
    if fresh:
        lines = list(fresh.values())
        detail = "\n".join("- " + s for s in lines[:20])
        if len(lines) > 20:
            detail += f"\n- {len(lines) - 20} additional events; see status.json."
        message = (f"Automated TransMLA campaign alert ({status['checked_at']}). "
                   "The user requested that this monitor wake this session on job events.\n"
                   f"{detail}\nCampaign: {bundle}\nStatus: {directory / 'status.json'}\n"
                   f"Slurm logs: {bundle / 'logs'}\n"
                   "Recheck live Slurm state and relevant logs/results, then report the outcome. "
                   "Investigate failures within the existing TransMLA task authorization. "
                   "Preserve validation gates and experiment scope. This monitor submits no jobs. "
                   "It remains installed; do not reinstall it or send another delivery test.")
        if progress:
            message += f"\nEvaluation progress: {progress['completed_chunks']}/{progress['total_chunks']} chunks complete. Pool logs: {progress['pool']}/logs."
        acknowledgement = send(config, message)
        with (directory / "alerts.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": now(), "events": fresh, "acknowledgement": acknowledgement}) + "\n")
        seen.update(fresh)
        status["delivered_events"] = sorted(seen)
        status["last_delivery"] = {"at": now(), "acknowledgement": acknowledgement}
        print(acknowledgement, flush=True)
    elif previous.get("last_delivery"):
        status["last_delivery"] = previous["last_delivery"]
    atomic_json(state_path, status)
    atomic_json(directory / "status.json", status)
    atomic_json(directory / "health.json", {"checked_at": now(), "status": "ok", "consecutive_errors": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--baseline", action="store_true", help="Acknowledge events already reported during setup")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = args.config.resolve().parent
    if (directory / "disabled").exists() and not args.preview:
        return
    with (directory / "poll.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            poll(config, directory, args.preview, args.baseline)
        except Exception as exc:
            if args.preview:
                raise
            path = directory / "health.json"
            prior = json.loads(path.read_text()) if path.exists() else {}
            failures = prior.get("consecutive_errors", 0) + 1
            health = {"checked_at": now(), "status": "error", "consecutive_errors": failures,
                      "error": str(exc), "alert_sent": prior.get("alert_sent", False)}
            print(json.dumps(health), flush=True)
            if failures >= 3 and not health["alert_sent"]:
                try:
                    health["acknowledgement"] = send(config, f"TransMLA monitor error after {failures} checks: "
                        f"{exc}. Inspect {path}. Do not infer job success from missing scheduler data.")
                    health["alert_sent"] = True
                except Exception as delivery_error:
                    health["delivery_error"] = str(delivery_error)
            atomic_json(path, health)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
