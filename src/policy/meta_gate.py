"""元门禁：审批"门禁自己能不能被改"。

普通门禁审候选模型；元门禁审候选门禁。
四条规则：
1. 只许收紧，不许放宽（放宽 = min_improvement 变小 / slice_tolerance 变大）；
2. 变更冷却：两次门禁变更之间至少隔 N 轮；
3. 锚定集：固定切片的误接受率不得上升；
4. 变更必须记录 diff，可回滚。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GateConfig:
    min_improvement: float = 1e-6
    slice_tolerance: float = 0.03

    def to_dict(self) -> dict:
        return asdict(self)


class MetaGate:
    def __init__(self, root, cooldown_rounds: int = 2):
        self.dir = Path(root) / "gates"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.cooldown = cooldown_rounds

    def _state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"last_change_round": -999, "history": []}

    def _save(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def check(self, old: GateConfig, new: GateConfig, *, round_no: int,
              anchored_false_accept: float, anchored_baseline: float) -> tuple:
        reasons = []
        if new.min_improvement < old.min_improvement:
            reasons.append(f"放宽了 min_improvement：{old.min_improvement} -> {new.min_improvement}")
        if new.slice_tolerance > old.slice_tolerance:
            reasons.append(f"放宽了 slice_tolerance：{old.slice_tolerance} -> {new.slice_tolerance}")
        state = self._state()
        if round_no - state["last_change_round"] < self.cooldown:
            reasons.append(f"冷却期未到：上次第 {state['last_change_round']} 轮，本次第 {round_no} 轮")
        if anchored_false_accept > anchored_baseline + 1e-9:
            reasons.append(f"锚定集误接受率上升：{anchored_baseline:.4f} -> {anchored_false_accept:.4f}")
        return (not reasons), reasons

    def check_thresholds(self, old: dict, new: dict, *, round_no: int,
                         anchored_false_accept: float, anchored_baseline: float) -> tuple:
        """策略阈值变更的元门禁（支持 {profile: {gap, spread}}）。

        阈值调高 = 更宽容 = 放宽，必须拒绝。
        """
        reasons = []
        profiles = [p for p in ("A", "B", "C") if p in old and p in new]
        for profile in profiles:
            if new[profile].get("gap", 0) > old[profile].get("gap", 0):
                reasons.append(f"放宽了 {profile}.gap：{old[profile]['gap']} -> {new[profile]['gap']}")
            if new[profile].get("spread", 0) > old[profile].get("spread", 0):
                reasons.append(f"放宽了 {profile}.spread：{old[profile]['spread']} -> {new[profile]['spread']}")
        state = self._state()
        if round_no - state["last_change_round"] < self.cooldown:
            reasons.append(f"冷却期未到：上次第 {state['last_change_round']} 轮，本次第 {round_no} 轮")
        if anchored_false_accept > anchored_baseline + 1e-9:
            reasons.append(f"锚定集误接受率上升：{anchored_baseline:.4f} -> {anchored_false_accept:.4f}")
        return (not reasons, reasons)

    def commit(self, old: GateConfig, new: GateConfig, *, round_no: int, note: str = "") -> Path:
        state = self._state()
        version = len(state["history"]) + 1
        payload = {
            "version": version,
            "round": round_no,
            "before": old.to_dict(),
            "after": new.to_dict(),
            "note": note,
        }
        (self.dir / f"g{version:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state["last_change_round"] = round_no
        state["history"].append(payload)
        self._save(state)
        return self.dir / f"g{version:03d}.json"

    def current(self) -> GateConfig:
        history = self._state()["history"]
        return GateConfig(**history[-1]["after"]) if history else GateConfig()
