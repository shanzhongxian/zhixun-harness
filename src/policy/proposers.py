"""提案器：产生“候选策略”，不负责采纳。

- BanditProposer：随机挑一个 profile，改它的 search 参数（lr/epochs/patience）
- CalibrateProposer：用每个 profile 自己的 gap 分位数校准阈值
"""
from __future__ import annotations

import copy
import random

from .spec import PROFILES, PolicySpec

LR_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]
EPOCH_GRID = [3, 5, 8]
PATIENCE_GRID = [1, 2, 3]


class BanditProposer:
    name = "bandit"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def propose(self, champion: PolicySpec, k: int = 3) -> list:
        out = []
        for _ in range(k):
            spec = copy.deepcopy(champion)
            spec.parent = champion.id
            spec.created_by = self.name
            profile = self.rng.choice(PROFILES)
            spec.search[profile]["lr_scale"] = self.rng.choice(LR_GRID)
            spec.search[profile]["epochs"] = self.rng.choice(EPOCH_GRID)
            spec.search[profile]["patience"] = self.rng.choice(PATIENCE_GRID)
            spec.replay = {"target_profile": profile}
            out.append(spec)
        return out


class CalibrateProposer:
    name = "calibrate"

    def __init__(self, quantile: float = 0.8, floor: float = 0.05):
        self.quantile = quantile
        self.floor = floor

    def propose(self, champion: PolicySpec, records: list) -> list:
        out = []
        for profile in PROFILES:
            gaps = sorted(r["diagnostics"].get("gap", 0.0) for r in records
                          if r.get("profile") == profile)
            if len(gaps) < 3:
                continue
            index = min(len(gaps) - 1, int(len(gaps) * self.quantile))
            threshold = round(max(self.floor, gaps[index]), 4)
            spec = copy.deepcopy(champion)
            spec.parent = champion.id
            spec.created_by = self.name
            spec.thresholds[profile]["gap"] = threshold
            spec.replay = {"target_profile": profile}
            out.append(spec)
        return out
