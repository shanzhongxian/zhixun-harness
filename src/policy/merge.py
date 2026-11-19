"""合并（#4）：只合数值型字段，且必须通过独立验证。

规则：
- 只合 thresholds / search（纯数值），不合 code / skills；
- 阈值取两者更严格的一侧（min gap、min spread），避免"合并=放宽"；
- 合并后单独跑回放，严格优于两个父版本才返回，否则返回 None + 原因。
"""
from __future__ import annotations

import copy

from .arena import score
from .spec import PROFILES, PolicySpec


def merge(a: PolicySpec, b: PolicySpec, records: list, min_gain: float = 1e-6) -> tuple:
    merged = copy.deepcopy(a)
    merged.parent = a.id
    merged.created_by = "merge"
    merged.skills = list(a.skills)
    merged.code = a.code

    for profile in PROFILES:
        for key in ("gap", "spread"):
            merged.thresholds[profile][key] = min(a.thresholds[profile][key], b.thresholds[profile][key])
        merged.search[profile] = dict(a.search[profile])

    sa, sb = score(a, records), score(b, records)
    sm = score(merged, records)
    best_parent = max(sa["score"], sb["score"])
    if sm["score"] > best_parent + min_gain:
        return merged, {"ok": True, "merged": sm, "a": sa, "b": sb}
    return None, {
        "ok": False,
        "reason": f"合并后得分 {sm['score']:.6f} 未超过最优父版 {best_parent:.6f}",
        "merged": sm, "a": sa, "b": sb,
    }
