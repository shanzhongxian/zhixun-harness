"""批循环：取批 → 读批次 → （可选训练）→ 交产物 → 等下一批。

默认只做「握手」：读取、校验、写 no_op 产物，不训练。
传 allow_train=True（handshake.py 加 --train）时，把批次数据喂进 harness 闭环：
诊断→决策→续训→门禁；accepted 写上线 artifact，否则写 no_op。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import project

from .artifact import write_artifact, write_no_op
from .batch import iter_batches, read_batch


def profile_config(profile: str) -> dict:
    package = project.PACKAGES[profile]
    return json.loads((package / "baselines" / "configs" / f"{profile}.json").read_text(encoding="utf-8"))


def _current_checkpoint(profile: str):
    """读版本指针（相对路径按仓库根解析），缺失时退回随包基线。"""
    import harness

    current = harness._load_current()
    return current.get(profile, {}).get("checkpoint")


def _train_one_batch(profile, data, info, submit_dir, round_no, strategy) -> dict:
    """把一个批次喂进 harness 闭环：诊断→决策→训练→门禁，再写产物。"""
    import harness
    from strategy.schemas import PRIMARY_METRIC, LR_SCALE_DEFAULTS

    primary = PRIMARY_METRIC[profile]
    record = harness.run_round(
        profile, strategy,
        lr_scale=LR_SCALE_DEFAULTS[profile], epochs=5, patience=2,
        batch_size=None, output_dir="runs/rounds", min_improvement=1e-6,
        model_id=os.environ.get("PYTRIO_MODEL", "Qwen/Qwen3.8-27B"),
        extra_train=data,
    )
    checkpoint = record.get("new_checkpoint")
    if record["result"] == "accepted" and checkpoint:
        candidate_metric = record["candidate_metric"]
        training_json = Path(checkpoint).parent / "training.json"
        target = write_artifact(
            submit_dir, profile, round_no, checkpoint,
            primary_metric=primary, value=candidate_metric[primary],
            gate="accepted", training_json=training_json if training_json.exists() else None,
            extra={"batch": info, "hypothesis": record["hypothesis"]},
        )
        deploy = True
    else:
        target = write_no_op(
            submit_dir, profile, round_no,
            current_checkpoint=_current_checkpoint(profile),
            hypothesis=record["hypothesis"], reason=record["result"],
            extra={"batch": info},
        )
        deploy = False
    return {
        "batch_id": info["batch_id"],
        "rows": info["n_rows"],
        "sha256": info["sha256"],
        "submit": project.to_store_path(target),
        "deploy": deploy,
        "result": record["result"],
        "action": record["action"],
    }


def run_batches(batch_dir, submit_dir, profile: str, *, strategy: str = "rule",
                allow_train: bool = False, limit: int | None = None) -> list[dict]:
    config = profile_config(profile)
    task = config["task"]
    classes = config.get("classes", [])
    expected_paths = None
    checkpoint = _current_checkpoint(profile)

    refs = iter_batches(batch_dir)
    if limit is not None:
        refs = refs[:limit]

    results = []
    for index, ref in enumerate(refs, 1):
        data, info = read_batch(ref, task, classes, expected_paths)
        expected_paths = data["paths"]  # 同一批次的特征 schema 必须一致

        if allow_train:
            result = _train_one_batch(profile, data, info, submit_dir, index, strategy)
        else:
            target = write_no_op(
                submit_dir, profile, index,
                current_checkpoint=checkpoint,
                hypothesis="T02 握手阶段：训练未开启，仅校验批次并占位",
                reason="dry_run",
                extra={"batch": info},
            )
            result = {
                "batch_id": info["batch_id"],
                "rows": info["n_rows"],
                "sha256": info["sha256"],
                "submit": project.to_store_path(target),
                "deploy": False,
                "result": "no_op",
                "action": "no_op",
            }
        results.append(result)
    return results
