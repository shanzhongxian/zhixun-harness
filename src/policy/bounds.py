"""策略边界：把 LLM/规则给出的参数夹到策略允许的范围内。

目的：策略文件不再是死代码，而是"可调范围的护栏"，
既保留 LLM 的判断，又不让它把 lr 调到 10 倍。
"""
from __future__ import annotations

LR_RATIO = (0.5, 1.5)      # 允许在策略值 ±50% 内浮动
EPOCH_RATIO = (0.5, 2.0)   # 允许 0.5x ~ 2x
PATIENCE_MAX_RATIO = 2.0
WD_MAX = 1e-3


def clamp_params(params: dict, search: dict) -> tuple:
    """返回 (夹取后的参数, 变更说明列表)。"""
    out = dict(params or {})
    notes = []

    if "lr_scale" in out and search.get("lr_scale"):
        base = float(search["lr_scale"])
        lo, hi = base * LR_RATIO[0], base * LR_RATIO[1]
        value = min(max(float(out["lr_scale"]), lo), hi)
        if abs(value - float(out["lr_scale"])) > 1e-12:
            notes.append(f"lr_scale {out['lr_scale']}→{round(value, 4)}（策略 {base}）")
        out["lr_scale"] = round(value, 4)

    if "epochs" in out and search.get("epochs"):
        base = int(search["epochs"])
        lo, hi = max(1, int(base * EPOCH_RATIO[0])), max(1, int(base * EPOCH_RATIO[1]))
        value = min(max(int(out["epochs"]), lo), hi)
        if value != int(out["epochs"]):
            notes.append(f"epochs {out['epochs']}→{value}（策略 {base}）")
        out["epochs"] = value

    if "patience" in out and search.get("patience"):
        base = int(search["patience"])
        hi = max(1, int(base * PATIENCE_MAX_RATIO))
        value = min(max(int(out["patience"]), 1), hi)
        if value != int(out["patience"]):
            notes.append(f"patience {out['patience']}→{value}（策略 {base}）")
        out["patience"] = value

    if "weight_decay" in out:
        value = min(max(float(out["weight_decay"]), 0.0), WD_MAX)
        if abs(value - float(out["weight_decay"])) > 1e-12:
            notes.append(f"weight_decay {out['weight_decay']}→{value}")
        out["weight_decay"] = value

    return out, notes
