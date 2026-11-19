"""训练师（策略层）与 harness 之间的接口契约。

这里定义三类东西：
1. 可调旋钮 schema（Qwen 能输出什么参数、范围、是否已接线）。
2. 观测报告字段约定（观测层按此产出，见 observability.py）。
3. 决策对象与校验（非法决策如何降级）。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

# 每个工况用于分组的 meta 字段。组 = 漂移 / 同一物理单元，绝不跨组随机切分。
GROUP_KEYS = {
    "A": ["source_batch"],          # 气体：按采集批次
    "B": ["bearing"],               # 轴承：按整根轴承
    "C": ["source_subset", "unit"], # 发动机：按子集 + 发动机编号
}

# 主指标与优化方向（1=越大越好，-1=越小越好）。与训练包 TASK.md 一致。
PRIMARY_METRIC = {"A": "macro_f1", "B": "rmse", "C": "rmse"}
DIRECTION = {"A": 1, "B": -1, "C": -1}

# 续训默认学习率倍率（相对随包默认学习率）。与 project.py 的 LR_SCALE_DEFAULTS 保持一致。
LR_SCALE_DEFAULTS = {"A": 0.3, "B": 0.5, "C": 0.3}

ACTIONS = ("train", "no_op", "rollback")

# 可调旋钮 schema：训练师只能输出这里声明的键。
# tier 1 = 第 1 版已接线；tier 2 = 只声明位置、尚未接线到训练代码（保留接口）。
PARAMS_SCHEMA = {
    "lr_scale":        {"tier": 1, "type": "float", "min": 0.01, "max": 1.0,  "default": None},
    "epochs":          {"tier": 1, "type": "int",   "min": 1,    "max": 200,  "default": 5},
    "batch_size":      {"tier": 1, "type": "int",   "min": 1,    "max": 65536, "default": None},
    "patience":        {"tier": 1, "type": "int",   "min": 1,    "max": 100,  "default": 2},
    "weight_decay":    {"tier": 1, "type": "float", "min": 0.0,  "max": 1e-3, "default": 0.0},
    "dropout":         {"tier": 1, "type": "float", "min": 0.0,  "max": 0.5,  "default": 0.0},
    "loss":            {"tier": 1, "type": "str",   "options": ["mse", "huber", "asymmetric"], "default": "mse"},
    "label_smoothing": {"tier": 2, "type": "float", "min": 0.0,  "max": 0.2,  "default": None},
    "class_weight":    {"tier": 2, "type": "str",   "options": ["balanced", "none"], "default": "none"},
}


@dataclass
class Decision:
    action: str
    hypothesis: str
    evidence_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "rule"  # rule | mock | qwen | fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "hypothesis": self.hypothesis,
            "evidence_ids": self.evidence_ids,
            "params": self.params,
            "source": self.source,
        }


def sanitize_params(params: Any) -> tuple:
    """只保留已启用(tier 1)且类型/范围合法的旋钮；越界夹到边界，非法则记入 errors。"""
    if not isinstance(params, dict):
        return {}, ["params 必须是对象"]
    clean, errors = {}, []
    for key, value in params.items():
        spec = PARAMS_SCHEMA.get(key)
        if spec is None:
            errors.append(f"未知旋钮 {key!r}")
            continue
        if spec["tier"] != 1:
            errors.append(f"旋钮 {key!r} 尚未接线")
            continue
        if spec["type"] == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{key} 必须是整数")
                continue
            clean[key] = max(spec["min"], min(spec["max"], value))
        elif spec["type"] == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{key} 必须是数字")
                continue
            clean[key] = max(spec["min"], min(spec["max"], float(value)))
        elif spec["type"] == "str":
            if not isinstance(value, str) or value not in spec.get("options", []):
                errors.append(f"{key} 必须是 {spec.get('options')} 之一")
                continue
            clean[key] = value
    return clean, errors


def validate_decision(decision: Any, profile: str) -> tuple:
    """校验并清洗一条决策。返回 (Decision, errors)。非法 action 默认回退为 no_op。"""
    errors: List[str] = []
    if not isinstance(decision, dict):
        return Decision("no_op", "", [], {}, "fallback"), ["decision 必须是对象"]
    action = decision.get("action")
    if action not in ACTIONS:
        errors.append(f"action 必须是 {ACTIONS} 之一，得到 {action!r}")
    hypothesis = decision.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        errors.append("hypothesis 必须是非空字符串")
    evidence = decision.get("evidence_ids", [])
    if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
        errors.append("evidence_ids 必须是字符串列表")
    params, param_errors = sanitize_params(decision.get("params", {}))
    errors.extend(param_errors)
    clean = Decision(
        action=action if action in ACTIONS else "no_op",
        hypothesis=hypothesis if isinstance(hypothesis, str) and hypothesis.strip() else "",
        evidence_ids=evidence if isinstance(evidence, list) else [],
        params=params,
        source=decision.get("source", "external"),
    )
    return clean, errors
