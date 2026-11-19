"""回放集：按分组确定性切分，用于检测历史/未见工况退化（防灾难性遗忘）。

回放集是"按时间/工况切分"的留出组，模拟评测时批次下发的未见工况。
切分只依赖组键排序，不依赖随机数，固定可复现。
"""
import numpy as np


def build_split(group_labels, *, holdout_frac=0.34):
    """把组的标签确定性地切成 gate / replay 两组。

    按组键字符串排序后，把末尾约 holdout_frac 的组划为 replay（未来/未见工况），
    其余为 gate。只有一个组时 replay=gate（如 A 单批次验证，退化检测退化为整体对比）。
    返回 {"gate": [...], "replay": [...]}，元素为组键元组。
    """
    unique = sorted(set(group_labels), key=str)
    if len(unique) <= 1:
        return {"gate": list(unique), "replay": list(unique)}
    k = max(1, min(len(unique) - 1, round(len(unique) * holdout_frac)))
    return {"gate": list(unique[:-k]), "replay": list(unique[-k:])}


def masks_for(group_labels, split):
    """返回 (gate_mask, replay_mask) 布尔数组，与验证行对齐。"""
    gate = set(split["gate"])
    replay = set(split["replay"])
    gate_mask = np.array([g in gate for g in group_labels])
    replay_mask = np.array([g in replay for g in group_labels])
    return gate_mask, replay_mask
