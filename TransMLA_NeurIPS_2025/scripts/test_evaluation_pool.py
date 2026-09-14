"""CPU-only tests for shared claims, failure isolation, and resume behavior."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import run_evaluation_pool as pool


class PoolTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / '.cache' / 'pool-tests'
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['locks', 'states', 'results']:
            (self.root / name).mkdir()

    def verify(self, index):
        if not (self.root / 'results' / str(index)).exists():
            raise ValueError('No result')

    def test_competing_workers_execute_each_chunk_once_and_resume(self):
        calls = []
        mutex = threading.Lock()
        def execute(index, fd):
            with mutex:
                calls.append(index)
            time.sleep(0.003)
            (self.root / 'results' / str(index)).write_text('done')
        with ThreadPoolExecutor(max_workers=8) as workers:
            outcomes = list(workers.map(lambda _: pool.drain(self.root, range(32), execute, self.verify), range(8)))
        self.assertEqual(outcomes, [0] * 8)
        self.assertEqual(sorted(calls), list(range(32)))
        pool.drain(self.root, range(32), execute, self.verify)
        self.assertEqual(len(calls), 32)

    def test_failed_chunk_does_not_block_other_work_or_retry_silently(self):
        calls = []
        def execute(index, fd):
            calls.append(index)
            if index == 1:
                raise RuntimeError('GPU error')
            (self.root / 'results' / str(index)).write_text('done')
        self.assertEqual(pool.drain(self.root, range(3), execute, self.verify), 1)
        self.assertEqual(pool.drain(self.root, range(3), execute, self.verify), 1)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(json.loads((self.root / 'states/0001.json').read_text())['status'], 'failed')

    def test_successful_exit_without_result_is_a_failure(self):
        self.assertEqual(pool.drain(self.root, [0], lambda i, fd: None, self.verify), 1)

    def test_interrupted_claim_can_be_reclaimed(self):
        pool.write_json(self.root / 'states/0000.json', {'status': 'running'})
        def execute(index, fd):
            (self.root / 'results' / str(index)).write_text('done')
        self.assertEqual(pool.drain(self.root, [0], execute, self.verify), 0)

    def test_missing_previously_completed_result_is_rejected(self):
        pool.write_json(self.root / 'states/0000.json', {'status': 'complete'})
        with self.assertRaisesRegex(ValueError, 'No result'):
            pool.drain(self.root, [0], lambda i, fd: None, self.verify)


if __name__ == '__main__':
    unittest.main()
