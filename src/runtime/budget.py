"""预算账本（T22）：CPU 分钟 / token / 标注配额三维，决策前拒绝、执行后结算。

只做账，不做决策；由 harness 在规划阶段调用 allow()，执行后调用 spend()。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Budget:
    limits: dict = field(default_factory=dict)
    spent: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ("gpu_minutes", "tokens", "labels"):
            self.limits.setdefault(key, 0.0)
            self.spent.setdefault(key, 0.0)

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        return cls(limits={
            "gpu_minutes": float(cfg.get("budget.gpu_minutes", 60.0)),
            "tokens": float(cfg.get("budget.tokens", 200000)),
            "labels": float(cfg.get("budget.labels", 500)),
        })

    def remaining(self) -> dict:
        return {k: max(0.0, self.limits[k] - self.spent[k]) for k in self.limits}

    def allow(self, cost: dict) -> tuple[bool, list]:
        """cost 里缺的维度按 0 计；任一维度不足即拒绝。"""
        reasons = []
        for key, limit in self.limits.items():
            need = float(cost.get(key, 0) or 0)
            if need < 0:
                reasons.append(f"{key} 成本不能为负")
            elif self.spent[key] + need > limit:
                reasons.append(f"{key} 不足：剩余 {self.limits[key] - self.spent[key]:.1f}，需要 {need:.1f}")
        return (not reasons, reasons)

    def spend(self, cost: dict) -> None:
        for key in self.spent:
            self.spent[key] += float(cost.get(key, 0) or 0)

    def snapshot(self) -> dict:
        return {"limits": dict(self.limits), "spent": dict(self.spent), "remaining": self.remaining()}

    def dump(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
