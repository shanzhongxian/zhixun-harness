"""效率策略：三个任务各用一种"减少参与训练的行"的方法，三种机制互不相同、均通用。

A -- dedup          内容去重：特征完全相同的行只保留第一条。气体识别是稳态采样，
                    重复行不携带新信息，去掉后物化、解析与训练同步变少。
B -- hash_sample    确定性抽样：按样本 id 的哈希取前 70%，保持原有分布与每个运行单元的覆盖，
                    不依赖标签，因此不会引入选择偏差。
C -- tail_stratified 尾部保留分层抽样：按标签分位切分，最高 10% 分位（高 RUL 尾部）
                    全量保留，其余压到总行数的 35%。尾部是最影响非对称评分的一段，
                    因此这一步只压缩稠密中段，不丢尾部。

三者都只决定"用哪些行继续训练"，不改网络结构、不改 checkpoint 格式，
因此仍然是从平台父检查点续训。
"""
from __future__ import annotations

import hashlib
import json

PROFILE_FILTER = {"A": "none", "B": "none", "C": "none"}
# v5 起 B/C 的行抽样默认关闭。v4 的平台实测：C 的尾部保留抽样把 MAE 从 −2.07% 变成 +12.67%
# （10 轮全部被拒、C 冻结在出厂模型），B 的 70% 抽样让隐藏集 RMSE 从 270.56 退化到 368.57。
# 因此只保留"解析提速 + A 的轮次级早停"这两个无损改动；抽样实现留着，需要时改回 "hash_sample" /
# "tail_stratified" 即可重新启用。
# A 用轮次级方法：数据只有 445 行，实测无重复、最近邻最小距离 0.30，无可压缩空间。
ABORT_PROFILES = {"A"}
ABORT_MARGIN = 0.02

# dedup
DEDUP_FIELDS = ("inputs",)
# hash_sample
HASH_KEEP_PERCENT = {"A": 100, "B": 70, "C": 100}
# tail_stratified
SAMPLE_RATIO = {"A": 1.0, "B": 1.0, "C": 0.35}
TAIL_QUANTILE = 0.90
# 末端 cooldown：最后一个 epoch 的学习率系数（train.py 使用）
COOLDOWN_FACTOR = 0.3


def filter_for(profile) -> str:
    return PROFILE_FILTER.get(str(profile).upper(), "none")


def abort_enabled(profile) -> bool:
    return str(profile).upper() in ABORT_PROFILES


def should_abort(direction, metric_before, metric_after, epochs_done):
    """只在第 1 个 epoch 之后判断：主指标相对退化超过 ABORT_MARGIN 就放弃本轮。"""
    if epochs_done != 1:
        return False, {}
    try:
        before = float(metric_before)
        after = float(metric_after)
    except (TypeError, ValueError):
        return False, {"reason": "metric unavailable"}
    if before == 0.0:
        return False, {"reason": "zero baseline"}
    gap = float(direction) * (after - before) / abs(before)
    info = {"strategy": "early_abort", "gap_relative": round(gap, 9),
            "margin": ABORT_MARGIN, "epochs_done": int(epochs_done)}
    return gap < -ABORT_MARGIN, info


def _quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(q)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _dedup(rows):
    seen = set()
    kept, dropped = [], 0
    for row in rows:
        signature = json.dumps({name: row.get(name) for name in DEDUP_FIELDS},
                               sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(row)
    return kept, {"strategy": "dedup", "dropped_duplicates": dropped}


def _hash_sample(rows, percent):
    if percent >= 100:
        return list(rows), {"strategy": "hash_sample", "applied": False}
    kept = []
    for row in rows:
        identity = str(row.get("id") or "")
        bucket = int(hashlib.md5(identity.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < percent:
            kept.append(row)
    return kept, {"strategy": "hash_sample", "applied": True, "keep_percent": percent}


def _tail_stratified(rows, ratio, tail_quantile):
    answers = []
    for row in rows:
        value = row.get("answer")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return list(rows), {"strategy": "tail_stratified", "applied": False,
                                "reason": "non-numeric target"}
        answers.append(float(value))
    if ratio >= 1.0 or not answers:
        return list(rows), {"strategy": "tail_stratified", "applied": False}
    cut = _quantile(answers, tail_quantile)
    tail = [row for row, value in zip(rows, answers) if value >= cut]
    rest = [row for row, value in zip(rows, answers) if value < cut]
    target = max(len(tail), int(round(len(rows) * ratio)))
    keep_rest = target - len(tail)
    if keep_rest >= len(rest) or keep_rest <= 0:
        return list(rows), {"strategy": "tail_stratified", "applied": False,
                            "reason": "tail covers target"}
    # 确定性等距抽样：不依赖随机数，同输入必得同输出，便于复现与审计。
    stride = len(rest) / float(keep_rest)
    picked = [rest[int(index * stride)] for index in range(keep_rest)]
    return picked + tail, {"strategy": "tail_stratified", "applied": True,
                           "tail_quantile": tail_quantile, "tail_rows_kept": len(tail),
                           "ratio": ratio}


def filter_rows(profile, rows):
    """按任务过滤训练行，返回 (rows, meta)。rows 是 id/inputs/answer 的字典型可迭代。"""
    rows = list(rows)
    before = len(rows)
    name = filter_for(profile)
    if not rows or name == "none":
        return rows, {"strategy": name, "applied": False, "rows_before": before, "rows_after": before}
    if name == "dedup":
        kept, meta = _dedup(rows)
    elif name == "hash_sample":
        kept, meta = _hash_sample(rows, HASH_KEEP_PERCENT.get(str(profile).upper(), 100))
    elif name == "tail_stratified":
        kept, meta = _tail_stratified(rows, SAMPLE_RATIO.get(str(profile).upper(), 1.0), TAIL_QUANTILE)
    else:
        kept, meta = rows, {"strategy": name, "applied": False}
    meta = {"rows_before": before, "rows_after": len(kept), **meta}
    if not kept:
        return rows, {"rows_before": before, "rows_after": before,
                      "strategy": name, "applied": False, "reason": "filter emptied training set"}
    return kept, meta
