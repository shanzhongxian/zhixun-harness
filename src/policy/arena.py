"""竞技场（Arena）：在历史决策点上“影子回放”一个策略，只读不训练。

按 profile 分别取阈值：A/B/C 的最优点不同，不能混用一套。
"""
from __future__ import annotations

import json
from pathlib import Path

from .spec import PolicySpec


def load_records(runs_dir) -> list:
    path = Path(runs_dir) / "decisions.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("diagnostics") and rec.get("candidate_metric"):
            records.append(rec)
    return records


def score(policy: PolicySpec, records: list) -> dict:
    """策略在这批历史决策点上的模拟收益（按 profile 取阈值，收益按基线归一化）。"""
    gains, skipped, missed = [], 0, 0
    for rec in records:
        profile = rec.get("profile")
        diag = rec["diagnostics"]
        if profile not in ("A", "B", "C"):
            continue
        gap = diag.get("gap", 0.0)
        if gap > policy.thresholds_for(profile)["gap"]:
            skipped += 1
            continue
        cand = rec["candidate_metric"]
        cur = diag.get("validation_metrics", {})
        primary = diag.get("primary_metric")
        direction = diag.get("direction", 1)
        if primary not in cand or primary not in cur:
            continue
        delta = direction * (cand[primary] - cur[primary])
        gain = delta / (abs(cur[primary]) or 1.0)
        if gain > 1e-9:
            gains.append(gain)
        else:
            missed += 1
    return {
        "score": round(sum(gains), 6),
        "accepted": len(gains),
        "rejected": missed,
        "skipped": skipped,
        "n": len(records),
    }


def compare(a: PolicySpec, b: PolicySpec, records: list) -> dict:
    sa, sb = score(a, records), score(b, records)
    return {"a": sa, "b": sb, "winner": a.id if sa["score"] >= sb["score"] else b.id,
            "margin": round(abs(sa["score"] - sb["score"]), 6)}
