"""Synthetic scheduler snapshots for the monitor; no scheduler or Codex writes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import monitor_campaign as monitor


class MonitorTests(unittest.TestCase):
    jobs = {"source": "100", "convert": "101", "diagnostic": "102", "full": "103"}
    counts = {"source": 2, "convert": 2, "diagnostic": 4, "full": 476}

    def test_compressed_arrays_and_steps(self):
        self.assertEqual(monitor.task_ids("103_[0-3,5-9:2%2]"),
                         ["103_0", "103_1", "103_2", "103_3", "103_5", "103_7", "103_9"])
        self.assertEqual(monitor.task_ids("100"), [])
        self.assertEqual(monitor.task_ids("100_0.batch"), [])

    def test_failure_and_dead_dependency_are_visible(self):
        records = monitor.parse_records("100_0|FAILED|1:0\n100_1|FAILED|1:0", 
            "101_[0-1%2]|PENDING|(DependencyNeverSatisfied)|afterok:100_*(failed)\n"
            "102_[0-3%2]|PENDING|(Dependency)|afterok:101_*\n"
            "103_[0-475%2]|PENDING|(Dependency)|afterok:102_*")
        summaries, events, terminal = monitor.summarize(self.jobs, self.counts, records)
        self.assertEqual(summaries["full"]["states"], {"PENDING": 476})
        self.assertEqual(summaries["source"]["states"], {"FAILED": 2})
        self.assertEqual(sum("blocked" in k for k in events), 1)
        self.assertFalse(terminal)

    def test_queue_absence_and_parent_success_do_not_imply_completion(self):
        records = monitor.parse_records("100|COMPLETED|0:0\n100_0|COMPLETED|0:0", "")
        summaries, events, terminal = monitor.summarize(self.jobs, self.counts, records)
        self.assertEqual(summaries["source"]["states"], {"COMPLETED": 1, "UNKNOWN": 1})
        self.assertFalse(any(":terminal:" in k for k in events))
        self.assertFalse(terminal)

    def test_live_requeue_overrides_stale_failure(self):
        records = monitor.parse_records("100_0|FAILED|1:0", "100_0|RUNNING|r001|null")
        self.assertEqual(records["100_0"]["state"], "RUNNING")

    def test_standalone_recovery_job_is_tracked_without_counting_array_parents(self):
        records = monitor.parse_records("100|COMPLETED|0:0\n200|FAILED|1:0", "")
        _, main_events, terminal = monitor.summarize(self.jobs, self.counts, records)
        summaries, events = monitor.auxiliary_events([{"job_id": "200", "label": "Cache diagnosis"}], records)
        self.assertFalse(terminal)
        self.assertNotIn("campaign:terminal", main_events)
        self.assertEqual(summaries["200"]["state"], "FAILED")
        self.assertEqual(list(events.values()), ["Cache diagnosis 200: FAILED"])

    def test_complete_campaign_requires_all_484_members(self):
        records = {f"{self.jobs[stage]}_{i}": {"state": "COMPLETED", "exit_code": "0:0"}
                   for stage, count in self.counts.items() for i in range(count)}
        self.assertEqual(len(records), 484)
        _, events, terminal = monitor.summarize(self.jobs, self.counts, records)
        self.assertTrue(terminal)
        self.assertIn("campaign:terminal", events)
        records["103_475"] = {"state": "CANCELLED", "exit_code": "0:15"}
        _, events, terminal = monitor.summarize(self.jobs, self.counts, records)
        self.assertTrue(terminal)
        self.assertTrue(any("103_475: CANCELLED" in v for v in events.values()))

    def test_pool_progress_preserves_completed_results_and_rejects_wrong_plan(self):
        cache = Path(__file__).resolve().parents[1] / '.cache' / 'monitor-tests'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            root = Path(temporary)
            (root / 'states').mkdir()
            (root / 'pool.json').write_text(json.dumps({'plan_sha256': 'abc', 'workers': 8,
                'indices': [28, 29], 'completed_before_pool': 28, 'total_chunks': 30}))
            (root / 'states/0028.json').write_text(json.dumps({'status': 'complete'}))
            (root / 'states/0029.json').write_text(json.dumps({'status': 'failed'}))
            override = {'pool': str(root), 'task_count': 8}
            progress = monitor.pool_progress(override, 'abc')
            self.assertEqual(progress['completed_chunks'], 29)
            self.assertEqual(progress['unfinished_chunks'], 1)
            self.assertEqual(progress['failed_chunks'], [29])
            with self.assertRaises(ValueError):
                monitor.pool_progress(override, 'wrong')

    def test_pool_override_counts_allocations_and_reports_unfinished_chunks(self):
        cache = Path(__file__).resolve().parents[1] / '.cache' / 'monitor-tests'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            root = Path(temporary)
            plan = {'jobs': {s: [None] * n for s, n in self.counts.items()}}
            plan_hash = monitor.hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            (root / 'plan.json').write_text(json.dumps(plan))
            (root / 'submission.json').write_text(json.dumps({'jobs': self.jobs, 'plan_sha256': plan_hash}))
            pool = root / 'pool'
            (pool / 'states').mkdir(parents=True)
            (pool / 'pool.json').write_text(json.dumps({'plan_sha256': plan_hash, 'workers': 2,
                'indices': list(range(28, 476)), 'completed_before_pool': 28, 'total_chunks': 476}))
            config = {'bundle': str(root), 'sacct': 'sacct', 'squeue': 'squeue', 'thread_id': 'test-thread',
                'interval_seconds': 300, 'accounting_start': '2026-09-09',
                'stage_overrides': {'full': {'job_id': '200', 'task_count': 2, 'pool': str(pool)}}}
            accounting = '\n'.join(f'{self.jobs[s]}_{i}|COMPLETED|0:0'
                for s in ['source', 'convert', 'diagnostic'] for i in range(self.counts[s]))
            accounting += '\n103_[28-475]|CANCELLED|0:0\n200_[0-1]|COMPLETED|0:0'
            with patch.object(monitor, 'command', side_effect=lambda args: accounting if args[0] == 'sacct' else ''), \
                    patch.object(monitor, 'send', return_value='Queued message test for thread test-thread.'):
                monitor.poll(config, root)
            status = json.loads((root / 'status.json').read_text())
            self.assertEqual(status['stages']['full']['expected_tasks'], 2)
            self.assertEqual(status['stages']['full']['states'], {'COMPLETED': 2})
            self.assertEqual(status['evaluation_progress']['completed_chunks'], 28)
            self.assertIn('pool:200:incomplete', status['delivered_events'])
            self.assertFalse(any('103_' in k for k in status['records']))

    def test_failed_delivery_retries_and_success_deduplicates(self):
        cache = Path(__file__).resolve().parents[1] / ".cache" / "monitor-tests"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            directory = Path(temporary)
            plan = {"jobs": {s: [None] * count for s, count in self.counts.items()}}
            (directory / "plan.json").write_text(json.dumps(plan, indent=2))
            (directory / "submission.json").write_text(json.dumps({"jobs": self.jobs,
                "plan_sha256": monitor.hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()}))
            config = {"bundle": str(directory), "sacct": "sacct", "squeue": "squeue",
                      "thread_id": "test-thread", "interval_seconds": 300, "accounting_start": "2026-09-09"}
            def scheduler(args):
                if args[0] == "sacct":
                    return "100_0|FAILED|1:0\n100_1|FAILED|1:0"
                return "\n".join(f"{self.jobs[s]}_[0-{self.counts[s]-1}%2]|PENDING|Dependency|afterok:100"
                                 for s in ("convert", "diagnostic", "full"))
            with patch.object(monitor, "command", side_effect=scheduler), patch.object(monitor, "send") as send:
                send.side_effect = RuntimeError("delivery unavailable")
                with self.assertRaises(RuntimeError):
                    monitor.poll(config, directory)
                self.assertFalse((directory / "state.json").exists())
                send.side_effect = None
                send.return_value = "Queued message test for thread test-thread."
                monitor.poll(config, directory)
                monitor.poll(config, directory)
                self.assertEqual(send.call_count, 2)
                self.assertEqual(len((directory / "alerts.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
