"""预算与兜底的小单测（T22/T13），不触发训练。

用法：python src/tests/test_runtime.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from runtime.budget import Budget  # noqa: E402
from runtime.deadline import Deadline  # noqa: E402
from runtime.output import atomic_write_json, write_fallback  # noqa: E402


class BudgetTest(unittest.TestCase):
    def test_allow_and_spend(self):
        b = Budget(limits={"gpu_minutes": 10, "tokens": 100, "labels": 5})
        ok, reasons = b.allow({"gpu_minutes": 3, "tokens": 50})
        self.assertTrue(ok, reasons)
        b.spend({"gpu_minutes": 3, "tokens": 50})
        self.assertEqual(b.remaining()["gpu_minutes"], 7)
        self.assertEqual(b.remaining()["tokens"], 50)

    def test_reject_over_limit(self):
        b = Budget(limits={"gpu_minutes": 2, "tokens": 10, "labels": 1})
        ok, reasons = b.allow({"gpu_minutes": 3})
        self.assertFalse(ok)
        self.assertTrue(any("gpu_minutes" in r for r in reasons))

    def test_missing_dimension_counts_zero(self):
        b = Budget(limits={"gpu_minutes": 1, "tokens": 1, "labels": 1})
        ok, _ = b.allow({})
        self.assertTrue(ok)
        self.assertEqual(b.remaining(), {"gpu_minutes": 1.0, "tokens": 1.0, "labels": 1.0})


class DeadlineTest(unittest.TestCase):
    def test_expire(self):
        d = Deadline(0.01)
        self.assertFalse(d.expired())
        d._end -= 1  # 手动把截止时间推到过去
        self.assertTrue(d.expired())
        with self.assertRaises(TimeoutError):
            d.check("test")


class OutputTest(unittest.TestCase):
    def test_fallback_result(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            path = write_fallback(out, "训练超时")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["reason"], "训练超时")
            atomic_write_json(out / "other.json", {"ok": True})
            self.assertFalse(list(out.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
