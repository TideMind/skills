from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts" / "context_kg_eval.py"
SPEC = importlib.util.spec_from_file_location("context_kg_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ContextKgEvalTest(unittest.TestCase):
    def test_fixture_meets_retrieval_targets(self) -> None:
        fixtures = Path(__file__).parent / "fixtures"
        report = MODULE.evaluate(
            fixtures / "context-kg", fixtures / "retrieval_cases.json"
        )

        self.assertEqual(1.0, report["recall_at_k"])
        self.assertEqual(1.0, report["mrr"])
        self.assertEqual(0.0, report["zero_hit_rate"])
        self.assertEqual([], report["failures"])


if __name__ == "__main__":
    unittest.main()
