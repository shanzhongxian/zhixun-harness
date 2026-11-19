"""训练型评估（方案 A）：让参数类候选"真训一小段"，用真实结果打分。

与 arena 的区别：
- arena 只在历史决策点上模拟"筛选"，换 lr/epochs 看不出差别；
- 这里从当前选中的 checkpoint 续训到临时目录，用公开验证集对比前后指标。

约束：
- 默认只跑 A（445 行，几秒钟）；B/C 可显式指定但更慢；
- 输出写到 .selftest/policy_probe/，不污染 runs/；
- 不更新版本指针（只有最终冠军才走 harness）。
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import project
from strategy.schemas import PRIMARY_METRIC


def _current_checkpoint(root: Path, profile: str) -> Path:
    current = json.loads((root / "runs" / "current.json").read_text(encoding="utf-8"))
    checkpoint = Path(current[profile]["checkpoint"])
    return checkpoint if checkpoint.is_absolute() else (root / checkpoint)


def _profile_config(profile: str) -> dict:
    path = project.PACKAGES[profile] / "baselines" / "configs" / f"{profile}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def probe(root, profile: str, *, lr_scale: float, epochs: int, patience: int = 2,
          weight_decay: float = 0.0, base: str = "current") -> dict:
    """base=current：从当前冠军续训；base=baseline：从随包基线重新探索。

    评分基准始终是"当前冠军的指标"，所以 baseline 模式回答的是
    "这组参数从零走一遍，能不能超过现在的冠军"。
    """
    root = Path(root)
    config = _profile_config(profile)
    primary = PRIMARY_METRIC[profile]
    direction = 1 if profile == "A" else -1

    champion_checkpoint = _current_checkpoint(root, profile)
    champion_model = project.Model.load(champion_checkpoint)
    validation = project.read_dataset(
        project.PACKAGES[profile] / "public" / "validation.jsonl",
        config["task"], config.get("classes", []), champion_model.meta["paths"])
    reference = project.evaluate(profile, champion_model, validation, nasa=(profile == "C"))[primary]

    start = champion_checkpoint if base == "current" else (project.PACKAGES[profile] / "checkpoint.npz")
    model = project.Model.load(start)
    before = project.evaluate(profile, model, validation, nasa=(profile == "C"))[primary]

    out = root / ".selftest" / "policy_probe" / f"{profile}-{uuid.uuid4().hex[:8]}"
    try:
        report = project.train_profile(
            profile, out, epochs=epochs, lr_scale=lr_scale, batch_size=None,
            patience=patience, resume=start, weight_decay=weight_decay)
        after_model = project.Model.load(out / "checkpoint.npz")
        after = project.evaluate(profile, after_model, validation, nasa=(profile == "C"))[primary]
    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)

    gain = direction * (after - reference) / (abs(reference) or 1.0)
    return {
        "score": round(gain, 6),
        "accepted": 1 if gain > 0 else 0,
        "rejected": 0 if gain > 0 else 1,
        "skipped": 0,
        "n": 1,
        "profile": profile,
        "base": base,
        "start": round(before, 6),
        "after": round(after, 6),
        "champion": round(reference, 6),
        "mode": "train_probe",
    }
