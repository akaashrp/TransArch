#!/usr/bin/env python3
"""Run frozen evaluation chunks in reusable GPU allocations, without changing model code.

Prepare performs CPU/artifact checks only. Worker claims are protected by flock;
each evaluation still uses the original campaign worker and validation gates.
No command in this script submits Slurm jobs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.writing')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def check_campaign(plan):
    from transmla.experiments.campaign import preflight
    from transmla.experiments.collect import collect
    from transmla.experiments.validate import require_gate
    preflight(plan)
    for row in plan['rows']:
        require_gate(row['gate'], row['model'])
        gate = Path(plan['root']) / 'gates' / f"{row['id']}-diagnostic.json"
        expected = {'status': 'passed', 'plan_sha256': digest(plan), 'row': row}
        if json.loads(gate.read_text()) != expected:
            raise ValueError(f'Diagnostic gate mismatch: {gate}')
    # Includes argument, environment, checkpoint and completed sample checks.
    return collect(plan)


def verify_result(plan, index):
    from transmla.experiments.collect import collect
    subset = {**plan, 'jobs': {**plan['jobs'], 'full': [plan['jobs']['full'][index]]}}
    if collect(subset)['missing_job_indices']:
        raise ValueError(f'Evaluation {index} exited without a complete result')


def prepare(plan_path, directory, workers):
    plan_path, directory = Path(plan_path).resolve(), Path(directory).resolve()
    if workers < 1 or directory.exists():
        raise ValueError('Use a fresh pool directory and a positive worker count')
    plan = json.loads(plan_path.read_text())
    report = check_campaign(plan)
    missing = report['missing_job_indices']
    manifest = {'plan': str(plan_path), 'plan_sha256': digest(plan),
                'indices': missing, 'completed_before_pool': len(plan['jobs']['full']) - len(missing),
                'total_chunks': len(plan['jobs']['full']), 'workers': workers,
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'prepared_at': now()}
    write_json(directory / 'pool.json', manifest)
    write_json(directory / 'preflight.json', {'status': 'passed', 'remaining': len(missing),
                                            'completed': manifest['completed_before_pool']})
    for name in ('locks', 'states', 'logs', 'workers'):
        (directory / name).mkdir()
    print(json.dumps(manifest, indent=2), flush=True)


def drain(directory, indices, execute, verify):
    """Execute each available chunk once; OS locks survive competing workers.

    A failed chunk is recorded and left for explicit recovery. Other chunks can
    finish. Interrupted chunks have no done marker and can be reclaimed after
    their previous processes release the lock. Existing outputs are verified.
    """
    directory = Path(directory)
    failures = 0
    for index in indices:
        state_path = directory / 'states' / f'{index:04d}.json'
        with (directory / 'locks' / f'{index:04d}.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            prior = json.loads(state_path.read_text()) if state_path.exists() else {}
            if prior.get('status') == 'failed':
                failures += 1
                continue
            if prior.get('status') == 'complete':
                verify(index)
                continue
            record = {'index': index, 'worker_job': os.environ.get('SLURM_JOB_ID'),
                      'worker_index': os.environ.get('SLURM_ARRAY_TASK_ID'), 'started_at': now()}
            write_json(state_path, {**record, 'status': 'running'})
            try:
                execute(index, lock.fileno())
                verify(index)
            except Exception as exc:
                failures += 1
                write_json(state_path, {**record, 'status': 'failed', 'ended_at': now(),
                                        'error': str(exc)})
                print(f'Chunk {index} FAILED: {exc}', flush=True)
            else:
                write_json(state_path, {**record, 'status': 'complete', 'ended_at': now()})
                print(f'Chunk {index} completed', flush=True)
    return failures


def run(pool_path):
    directory = Path(pool_path).resolve().parent
    manifest = json.loads(Path(pool_path).read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest['runner_sha256']:
        raise ValueError('Pool runner changed since preparation')
    plan = json.loads(Path(manifest['plan']).read_text())
    if digest(plan) != manifest['plan_sha256']:
        raise ValueError('Frozen campaign plan changed')
    report = check_campaign(plan)
    missing = set(report['missing_job_indices'])
    child = None

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def execute(index, lock_fd):
        nonlocal child
        if index not in missing:
            return  # check_campaign already verified this completed result.
        command = [sys.executable, '-m', 'transmla.experiments.campaign', 'worker',
                   '--plan', manifest['plan'], '--stage', 'full', '--index', str(index)]
        print(f'Starting chunk {index}: {plan["jobs"]["full"][index]["row"]["id"]}', flush=True)
        with (directory / 'logs' / f'chunk-{index:04d}.out').open('a') as log:
            # The child also holds the claim if its supervising process dies.
            child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(lock_fd,))
            code = child.wait()
            child = None
        if code:
            raise RuntimeError(f'Original campaign worker exited {code}; see chunk-{index:04d}.out')

    failures = drain(directory, manifest['indices'], execute, lambda i: verify_result(plan, i))
    worker_id = f'{os.environ.get("SLURM_JOB_ID", "local")}-{os.environ.get("SLURM_ARRAY_TASK_ID", os.getpid())}'
    write_json(directory / 'workers' / f'{worker_id}.json',
               {'ended_at': now(), 'status': 'failed' if failures else 'finished', 'failures': failures})
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('worker')
    p.add_argument('--pool', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.plan, args.out, args.workers)
        return 0
    return run(args.pool)


if __name__ == '__main__':
    raise SystemExit(main())
