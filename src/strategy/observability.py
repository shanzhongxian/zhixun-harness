"""观测层：把模型与数据翻译成训练师（Qwen）能读的数字化诊断。

训练师是个 LLM，只能"看见"文本。这里的职责是把过拟合 / 漂移 / 最差组等
信号预计算成命名、数字化的事实，而不是让 LLM 自己从原始数字里推断。
"""
import json
from pathlib import Path

import numpy as np

from .schemas import GROUP_KEYS, PRIMARY_METRIC, DIRECTION

# 第一版启发式阈值（后续用统计检验替换）。gap = val_loss - train_loss。
_GAP_THRESHOLD = {"A": 0.6, "B": 0.7, "C": 0.7}
# 分组主指标的相对离散度阈值（漂移信号）。相对离散度 = (max-min)/mean。
_SPREAD_THRESHOLD = {"A": 0.5, "B": 0.4, "C": 0.3}


def extract_group(meta: dict, profile: str):
    """从一行 meta 提取分组键元组。"""
    return tuple(meta.get(key) for key in GROUP_KEYS[profile])


def load_meta(path) -> list:
    """按行读回 JSONL 的 meta 字段，顺序与 data.read_dataset 一致（跳过空行）。"""
    metas = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metas.append(json.loads(line).get("meta", {}))
    return metas


def _loss(model, x, y) -> float:
    """复现 fit 的逐样本平均损失（在归一化目标空间计算），用于训练/验证差距。"""
    task = model.meta["task"]
    xn = model.normalized(x)
    if task == "regression":
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
        _, logits = model.forward(xn)
        err = logits[:, 0] - target
        return float(np.mean(err ** 2))
    _, logits = model.forward(xn)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-300)).mean())


def _group_primary(profile, model, x, y) -> float:
    """按组计算主指标（分类=macro_f1，回归=rmse），不附带 NASA 分。"""
    pred = model.predict(x)
    if model.meta["task"] == "classification":
        labels = pred.argmax(axis=1)
        observed = sorted(set(y.tolist()) | set(labels.tolist()))
        f1 = [2 * np.sum((labels == c) & (y == c)) / (np.sum(labels == c) + np.sum(y == c))
              for c in observed]
        return float(np.mean(f1))
    err = pred - y
    return float(np.sqrt(np.mean(err ** 2)))


def compute_diagnostics(profile, model, train, validation, evaluate, validation_meta,
                        thresholds=None) -> dict:
    """生成观测报告。evaluate(data) 返回该 profile 的完整指标字典。

    thresholds: {"gap": x, "spread": y}，来自冠军策略；缺省回退到常量。
    """
    primary = PRIMARY_METRIC[profile]
    gap_threshold = float((thresholds or {}).get("gap", _GAP_THRESHOLD[profile]))
    spread_threshold = float((thresholds or {}).get("spread", _SPREAD_THRESHOLD[profile]))
    train_metrics = evaluate(train, False)
    validation_metrics = evaluate(validation, profile == "C")

    train_loss = _loss(model, train["x"], train["y"])
    validation_loss = _loss(model, validation["x"], validation["y"])
    gap = validation_loss - train_loss

    groups = [extract_group(m, profile) for m in validation_meta]
    group_metrics = []
    for group in sorted(set(groups), key=lambda g: str(g)):
        mask = np.array([g == group for g in groups])
        if mask.sum() == 0:
            continue
        group_metrics.append({
            "group": list(group),
            "primary": _group_primary(profile, model, validation["x"][mask], validation["y"][mask]),
            "n": int(mask.sum()),
        })

    overfit_flag = gap > gap_threshold
    drift_flag = False
    if len(group_metrics) >= 2:
        values = [g["primary"] for g in group_metrics]
        mean = abs(float(np.mean(values))) + 1e-9
        spread = (max(values) - min(values)) / mean
        drift_flag = spread > spread_threshold

    return {
        "profile": profile,
        "primary_metric": primary,
        "direction": DIRECTION[profile],
        "thresholds_used": {"gap": gap_threshold, "spread": spread_threshold},
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "gap": gap,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "group_metrics": group_metrics,
        "n_groups": len(group_metrics),
        "overfit_flag": bool(overfit_flag),
        "drift_flag": bool(drift_flag),
    }
