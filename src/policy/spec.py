"""策略对象（PolicySpec）：按 profile 分开存，可版本化。

设计：
- thresholds / search 都是 {A: {...}, B: {...}, C: {...}}，
  因为三个任务的最优点本来不同（A lr 0.3 / B 0.5 / C 0.3）；
- 兼容旧格式：读到扁平 dict 时自动铺到三个 profile；
- 改策略 = 生成新版本，不改代码。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

PROFILES = ("A", "B", "C")

DEFAULT_THRESHOLDS = {
    "A": {"gap": 0.6, "spread": 0.5},
    "B": {"gap": 0.7, "spread": 0.4},
    "C": {"gap": 0.7, "spread": 0.3},
}
DEFAULT_SEARCH = {
    "A": {"lr_scale": 0.3, "epochs": 5, "patience": 2, "weight_decay": 0.0},
    "B": {"lr_scale": 0.5, "epochs": 5, "patience": 2, "weight_decay": 0.0},
    "C": {"lr_scale": 0.3, "epochs": 5, "patience": 2, "weight_decay": 0.0},
}
DEFAULT_ROUTER = {"rule": 0.2, "qwen": 0.8, "skill": 0.0}


def _per_profile(value: dict, default: dict) -> dict:
    """扁平 dict 自动铺到三个 profile；已是分 profile 结构则原样用。"""
    if set(PROFILES) <= set(value) and all(isinstance(value[p], dict) for p in PROFILES):
        return {p: {**default[p], **value[p]} for p in PROFILES}
    return {p: {**default[p], **value} for p in PROFILES}


@dataclass
class PolicySpec:
    id: str
    parent: str | None = None
    created_by: str = "bootstrap"
    thresholds: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_THRESHOLDS))
    search: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_SEARCH))
    router: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_ROUTER))
    skills: list = field(default_factory=list)
    code: str | None = None
    replay: dict = field(default_factory=dict)
    status: str = "challenger"

    def __post_init__(self):
        self.thresholds = _per_profile(self.thresholds, DEFAULT_THRESHOLDS)
        self.search = _per_profile(self.search, DEFAULT_SEARCH)

    def thresholds_for(self, profile: str) -> dict:
        return self.thresholds[profile]

    def search_for(self, profile: str) -> dict:
        return self.search[profile]

    def validate(self) -> list:
        errors = []
        for profile in PROFILES:
            t, s = self.thresholds[profile], self.search[profile]
            if not 0 < t.get("gap", 0) <= 5:
                errors.append(f"{profile}.thresholds.gap 需在 (0,5]")
            if not 0 < t.get("spread", 0) <= 5:
                errors.append(f"{profile}.thresholds.spread 需在 (0,5]")
            if not 0.01 <= s.get("lr_scale", 0) <= 1.0:
                errors.append(f"{profile}.search.lr_scale 需在 [0.01,1]")
            if not 1 <= s.get("epochs", 0) <= 200:
                errors.append(f"{profile}.search.epochs 需在 [1,200]")
            if not 1 <= s.get("patience", 0) <= 100:
                errors.append(f"{profile}.search.patience 需在 [1,100]")
            if not 0 <= s.get("weight_decay", 0) <= 1e-3:
                errors.append(f"{profile}.search.weight_decay 需在 [0,1e-3]")
        total = sum(self.router.values())
        if abs(total - 1.0) > 1e-6:
            errors.append(f"router 权重和应为 1，实际 {total}")
        return errors

    def loose_fields(self, other: "PolicySpec") -> list:
        """阈值调高 = 更宽容 = 放宽；按 profile 逐个比较。"""
        loose = []
        for profile in PROFILES:
            if self.thresholds[profile]["gap"] > other.thresholds[profile]["gap"]:
                loose.append(f"{profile}.gap")
            if self.thresholds[profile]["spread"] > other.thresholds[profile]["spread"]:
                loose.append(f"{profile}.spread")
        return loose

    def diff(self, other: "PolicySpec") -> dict:
        changes = {}
        for name in ("thresholds", "search", "router", "skills", "code", "status"):
            a, b = getattr(self, name), getattr(other, name)
            if a != b:
                changes[name] = {"from": b, "to": a}
        return changes

    def summary(self) -> dict:
        """给 SwanLab / 记录用的轻量摘要（不含 skills/code 全文）。"""
        return {
            "id": self.id,
            "parent": self.parent,
            "created_by": self.created_by,
            "thresholds": self.thresholds,
            "search": self.search,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id, "parent": self.parent, "created_by": self.created_by,
            "thresholds": self.thresholds, "search": self.search, "router": self.router,
            "skills": self.skills, "code": self.code, "replay": self.replay, "status": self.status,
        }

    def save(self, path):
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "PolicySpec":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    @classmethod
    def load(cls, path) -> "PolicySpec":
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
