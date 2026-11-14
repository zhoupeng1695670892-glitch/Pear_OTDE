from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import unittest
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

class RecordedResult(unittest.TextTestResult):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({'test': test.id(), 'status': 'passed'})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({'test': test.id(), 'status': 'failed', 'detail': self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({'test': test.id(), 'status': 'error', 'detail': self._exc_info_to_string(err, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({'test': test.id(), 'status': 'skipped', 'reason': reason})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'docs' / 'validation_results.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'module mathematics/numerics and adapter contract tests; not full native Master training', 'python': platform.python_version(), 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cpu_threads': 2, 'tests_run': result.testsRun, 'passed': sum((r['status'] == 'passed' for r in result.records)), 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'elapsed_seconds': time.perf_counter() - start, 'tests': result.records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
