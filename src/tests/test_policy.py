"""策略竞技场单测：spec / store / arena / meta_gate / merge / 双包隔离。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from data.source import DualSource  # noqa: E402
from policy.arena import score  # noqa: E402
from policy.merge import merge  # noqa: E402
from policy.meta_gate import MetaGate  # noqa: E402
from policy.spec import PolicySpec  # noqa: E402
from policy.store import PolicyStore  # noqa: E402


def rec(profile, gap, cand, cur, primary="macro_f1", direction=1):
    return {
        "profile": profile,
        "diagnostics": {"gap": gap, "primary_metric": primary, "direction": direction,
                        "validation_metrics": {primary: cur}},
        "candidate_metric": {primary: cand},
    }


RECORDS = [
    rec("A", 0.10, 0.720, 0.710),   # 提升，应采纳
    rec("A", 0.20, 0.705, 0.710),   # 退化，应拒绝
    rec("A", 0.70, 0.900, 0.710),   # gap 超阈值，应跳过
]


class SpecTest(unittest.TestCase):
    def test_per_profile_defaults(self):
        spec = PolicySpec(id="p001")
        self.assertEqual(spec.thresholds_for("A")["gap"], 0.6)
        self.assertEqual(spec.thresholds_for("B")["gap"], 0.7)
        self.assertEqual(spec.search_for("B")["lr_scale"], 0.5)
        self.assertEqual(spec.validate(), [])

    def test_legacy_flat_dict_migrated(self):
        spec = PolicySpec.from_dict({"id": "p9", "thresholds": {"gap": 0.4, "spread": 0.2}})
        for profile in ("A", "B", "C"):
            self.assertEqual(spec.thresholds_for(profile)["gap"], 0.4)

    def test_loose_detection_per_profile(self):
        a, b = PolicySpec(id="a"), PolicySpec(id="b")
        b.thresholds["B"]["gap"] = 0.9
        self.assertIn("B.gap", b.loose_fields(a))
        b.thresholds["B"]["gap"] = 0.3
        self.assertNotIn("B.gap", b.loose_fields(a))

    def test_invalid_params(self):
        spec = PolicySpec(id="x")
        spec.search["A"]["epochs"] = 0
        self.assertTrue(spec.validate())


class StoreTest(unittest.TestCase):
    def test_commit_and_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            store = PolicyStore(temp)
            p1 = PolicySpec(id=store.next_id())
            store.commit(p1)
            p2 = PolicySpec(id=store.next_id(), parent=p1.id)
            store.commit(p2)
            self.assertEqual(store.champion().id, p2.id)
            store.rollback(p1.id)
            self.assertEqual(store.champion().id, p1.id)


class ArenaTest(unittest.TestCase):
    def test_score_respects_threshold(self):
        stat = score(PolicySpec(id="p1"), RECORDS)
        self.assertEqual(stat["skipped"], 1)
        self.assertEqual(stat["accepted"], 1)
        self.assertAlmostEqual(stat["score"], 0.014085, places=5)

    def test_profile_thresholds_are_independent(self):
        spec = PolicySpec(id="p2")
        spec.thresholds["A"]["gap"] = 0.05      # A 收紧，全部跳过
        self.assertEqual(score(spec, RECORDS)["accepted"], 0)
        self.assertEqual(score(spec, RECORDS)["skipped"], 3)


class MetaGateTest(unittest.TestCase):
    def test_reject_loosening(self):
        with tempfile.TemporaryDirectory() as temp:
            gate = MetaGate(temp, cooldown_rounds=2)
            old = {"A": {"gap": 0.6, "spread": 0.5}}
            new = {"A": {"gap": 0.8, "spread": 0.5}}
            ok, reasons = gate.check_thresholds(old, new, round_no=5,
                                                anchored_false_accept=0.1, anchored_baseline=0.1)
            self.assertFalse(ok)
            self.assertTrue(any("A.gap" in r for r in reasons))

    def test_accept_tightening(self):
        with tempfile.TemporaryDirectory() as temp:
            gate = MetaGate(temp, cooldown_rounds=2)
            old = {"A": {"gap": 0.6, "spread": 0.5}}
            new = {"A": {"gap": 0.4, "spread": 0.3}}
            ok, reasons = gate.check_thresholds(old, new, round_no=5,
                                                anchored_false_accept=0.1, anchored_baseline=0.1)
            self.assertTrue(ok, reasons)


class MergeTest(unittest.TestCase):
    def test_reject_when_not_better(self):
        merged, info = merge(PolicySpec(id="a"), PolicySpec(id="b"), RECORDS)
        self.assertIsNone(merged)
        self.assertIn("未超过", info["reason"])


class BoundsTest(unittest.TestCase):
    def test_clamp_extremes(self):
        from policy.bounds import clamp_params
        search = {"lr_scale": 0.3, "epochs": 5, "patience": 2}
        out, notes = clamp_params({"lr_scale": 5.0, "epochs": 100, "patience": 50}, search)
        self.assertAlmostEqual(out["lr_scale"], 0.45, places=4)   # 0.3 × 1.5
        self.assertEqual(out["epochs"], 10)                        # 5 × 2
        self.assertEqual(out["patience"], 4)                       # 2 × 2
        self.assertEqual(len(notes), 3)

    def test_clamp_noop_when_inside(self):
        from policy.bounds import clamp_params
        search = {"lr_scale": 0.3, "epochs": 5, "patience": 2}
        out, notes = clamp_params({"lr_scale": 0.32, "epochs": 6, "patience": 2}, search)
        self.assertEqual(notes, [])
        self.assertEqual(out["epochs"], 6)


class DualSourceTest(unittest.TestCase):
    def test_isolation_and_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            dual = DualSource(temp)
            dual.prepare()
            plan = dual.unified_plan("p001")
            self.assertEqual([s["name"] for s in plan["sources"]], ["A", "B"])
            with self.assertRaises(ValueError):
                dual.A.resolve("../B/inbox")


if __name__ == "__main__":
    unittest.main(verbosity=2)
