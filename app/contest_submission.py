"""参赛提交入口（contest.harness.workspace.v2）。

describe: 由 SDK 输出协议行。
train:   读公开数据 → 走平台网关做一次决策 → 继续训练基线权重 → 提交候选 → 收尾。
predict: 逐行读 {task, inputs}，用本轮产物作答。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
if (BASE / "baselines").is_dir() and str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import contest_data  # noqa: E402
import contest_llm  # noqa: E402
import contest_paths as paths  # noqa: E402

TASKS = ["classification", "regression"]
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
_METRIC_DIRECTION = {"macro_f1": "higher", "mae": "lower", "rmse": "lower"}
# 主指标之外的夹具指标：主指标好转但夹具指标大幅变差时否决，避免"最差阶段"被拉垮。
_SECONDARY_GUARD = {
    "macro_f1": {"accuracy": 0.01},
    "mae": {"rmse": 0.02},
    "rmse": {"mae": 0.02},
}
_HIGHER = ("macro_f1", "accuracy")
# 单轮软截止占平台墙钟预算的比例；只在 wall_seconds 可用且不太小时生效。
_BUDGET_FRACTION = 0.6
_MIN_BUDGET_SECONDS = 60.0


def _log(message: str) -> None:
    print(str(message), file=sys.stderr, flush=True)


def _sha256(path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config_path(key: str) -> Path:
    candidate = BASE / "baselines" / "configs" / ("%s.json" % key)
    if not candidate.is_file():
        raise RuntimeError("缺少配置文件：%s" % candidate)
    return candidate


def _checkpoint_path(_key: str = "") -> Path:
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH", "") or "")
    if path and path.is_dir():
        path = path / "checkpoint.npz"
    if not path:
        path = paths.model_dir() / "checkpoint.npz"
    if not path.is_file():
        raise RuntimeError("找不到本轮产物：%s" % path)
    return path


def _parent_checkpoint(rnd) -> Path:
    declared = list(rnd.parent.descriptor.get("checkpoint_paths") or [])
    if not declared:
        raise RuntimeError("父代产物没有声明 checkpoint")
    return rnd.parent.path(declared[0], root=paths.previous_dir())


def _materialize(rnd, work: Path, key: str):
    from baselines import efficiency  # 延迟导入，describe 阶段不加载
    train_path = work / "train.jsonl"
    validation_path = work / "validation.jsonl"
    # 效率策略在物化阶段生效：少写、少解析、少训练，三者同源。
    rows, filter_meta = efficiency.filter_rows(key, rnd.data.training_rows())
    train_rows = contest_data.write_jsonl(rows, train_path)
    validation_rows = contest_data.write_jsonl(rnd.data.validation_rows(), validation_path)
    if not train_rows or not validation_rows:
        raise RuntimeError("公开数据为空：train=%d validation=%d" % (train_rows, validation_rows))
    return train_path, validation_path, train_rows, validation_rows, filter_meta


def _summarize(rnd, key: str, history, train_rows: int, validation_rows: int) -> dict:
    contract = rnd.task if isinstance(rnd.task, dict) else {}
    return {
        "dataset": key,
        "task": contract.get("task", "classification"),
        "metric": contract.get("metric", "macro_f1"),
        "metric_direction": contract.get("metric_direction", "higher"),
        "round_index": rnd.round_index,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "history": [
            {"id": item.get("id"), "dataset": item.get("dataset_key"), "round": item.get("round_index"),
             "hypothesis": (item.get("hypothesis") or "")[:200]}
            for item in history[-5:]
        ],
        "budget": {"wall_seconds": rnd.wall_seconds, "max_output_tokens": rnd.max_output_tokens},
    }


def _provenance(work: Path) -> Path:
    target = work / "label-provenance.json"
    target.write_text(json.dumps({
        "status": "approved",
        "method": "使用平台公开训练/验证集自带的 answer 字段作为 RUL 标签，不做任何派生或推断。",
    }, ensure_ascii=False), encoding="utf-8")
    return target


def _evaluate(metric: str, before, after, tolerance: float = 0.0, direction: str = "") -> dict:
    # 方向优先用平台任务合同里的 metric_direction（权威），映射表只作兜底；
    # 两者都拿不到时才退回 "higher"，并在返回值里标出，便于事后核对。
    resolved = str(direction or "").strip().lower()
    source = "contract"
    if resolved not in ("higher", "lower"):
        resolved = _METRIC_DIRECTION.get(str(metric).strip().lower(), "")
        source = "map"
    if resolved not in ("higher", "lower"):
        resolved = "higher"
        source = "default"
    before = dict(before or {})
    after = dict(after or {})
    value_before = before.get(metric)
    value_after = after.get(metric)
    improved = False
    if _number(value_before) and _number(value_after):
        # tolerance 允许相对退化：阶段切换时跨阶段比较没有参照，留 0.5% 余量。
        slack = abs(float(value_before)) * float(tolerance)
        if resolved == "higher":
            improved = value_after > value_before - slack
        else:
            improved = value_after < value_before + slack
    return {"metric": metric, "direction": resolved, "direction_source": source,
            "before": value_before, "after": value_after,
            "tolerance": float(tolerance), "improved": bool(improved)}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clip(text: str, limit: int = 3800) -> str:
    """按 UTF-8 字节截断：SDK 的 4000 上限是字节数，不是字符数。"""
    raw = str(text)
    if len(raw.encode("utf-8")) <= limit:
        return raw
    out = []
    used = 0
    for char in raw:
        size = len(char.encode("utf-8"))
        if used + size > limit:
            break
        out.append(char)
        used += size
    return "".join(out)


def _guard(metric: str, before, after) -> dict:
    """夹具指标否决：主指标之外，任一夹具指标退化超过容差就否决本轮候选。"""
    rules = _SECONDARY_GUARD.get(metric) or {}
    notes = []
    vetoed = False
    before = dict(before or {})
    after = dict(after or {})
    for name, tolerance in rules.items():
        low, high = before.get(name), after.get(name)
        if not _number(low) or not _number(high):
            continue
        if name in _HIGHER:
            degraded = high < low - tolerance
        else:
            degraded = bool(low) and high > low * (1.0 + tolerance)
        notes.append({"metric": name, "before": low, "after": high,
                      "tolerance": tolerance, "degraded": bool(degraded)})
        if degraded:
            vetoed = True
    return {"secondary": notes, "vetoed": vetoed}


def _rejections(rounds) -> int:
    """连续未采纳轮数：本轮决策据此强制换方向。"""
    count = 0
    for item in reversed(list(rounds or [])):
        if not isinstance(item, dict):
            break
        if item.get("accepted") or item.get("improved"):
            break
        count += 1
    return count


def _deadline(rnd, started: float):
    """平台给了墙钟预算时才启用软截止；只防超时，不会让本轮没有产物。"""
    budget = getattr(rnd, "wall_seconds", None)
    if not _number(budget) or float(budget) < _MIN_BUDGET_SECONDS:
        return None
    return started + max(_MIN_BUDGET_SECONDS, float(budget) * _BUDGET_FRACTION)


def _gap(metric: str, before, after):
    """本轮相对父检查点的提升量（正数=变好），用于写入 state 供下一轮决策。"""
    try:
        low = float(dict(before or {})[metric])
        high = float(dict(after or {})[metric])
    except (KeyError, TypeError, ValueError):
        return None
    direction = _METRIC_DIRECTION.get(metric, "higher")
    return round(high - low, 9) if direction == "higher" else round(low - high, 9)


def _recon(rnd, key: str, memory: dict) -> None:
    """把平台给的可见字段打到 stderr（stdout 只留协议帧）。"""
    def value(name, default=None):
        return getattr(rnd, name, default)

    _log("[recon] dataset=%s round=%s stage=%r wall_seconds=%r max_output_tokens=%r artifacts=%d "
         "state=%s memory_rounds=%d metric=%s"
         % (key, value("round_index"), value("stage_id", ""), value("wall_seconds"),
            value("max_output_tokens"), len(value("artifacts", []) or []),
            bool(value("state_path")), len(memory.get("rounds") or []),
            (value("task", {}) or {}).get("metric") if isinstance(value("task", {}), dict) else None))
    limits = value("limits", {}) or {}
    if limits:
        _log("[recon] limits=%s" % json.dumps(limits, ensure_ascii=False, sort_keys=True)[:400])


def _load_memory(rnd, work: Path) -> dict:
    """把上一轮的状态目录搬到可写区再读；没有状态就是第一次迭代。"""
    target = work / "state-in"
    try:
        rnd.state_workspace(target)
    except Exception as error:  # 无历史状态或平台未挂载
        _log("state_workspace skipped: %s" % type(error).__name__)
        return {"rounds": []}
    memory_path = target / "memory.json"
    if not memory_path.is_file():
        return {"rounds": []}
    try:
        value = json.loads(memory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"rounds": []}
    rounds = value.get("rounds") if isinstance(value, dict) else None
    return {"rounds": rounds if isinstance(rounds, list) else []}


def train(rnd) -> None:
    round_started = time.monotonic()
    key = contest_data.dataset_key(rnd.dataset_key)
    work = paths.workspace() / ("round-%02d-%s" % (rnd.round_index, key))
    index = 0
    while work.exists():
        index += 1
        work = paths.workspace() / ("round-%02d-%s-%d" % (rnd.round_index, key, index))
    work.mkdir(parents=True)
    memory = _load_memory(rnd, work)
    _recon(rnd, key, memory)
    train_path, validation_path, train_rows, validation_rows, filter_meta = _materialize(rnd, work, key)
    _log("[timing] materialize=%.2fs rows=%d/%d filter=%s"
         % (time.monotonic() - round_started, train_rows, validation_rows,
            json.dumps(filter_meta, ensure_ascii=False)))
    parent = _parent_checkpoint(rnd)
    history = rnd.history()
    config = _config_path(key)
    config_body = json.loads(config.read_text(encoding="utf-8"))
    baseline_lr = float(config_body["learning_rate"])

    context = _summarize(rnd, key, history, train_rows, validation_rows)
    context["parent"] = {"artifact_id": rnd.parent.artifact_id, "checkpoint": parent.name,
                         "baseline_learning_rate": baseline_lr, "baseline_epochs": config_body["epochs"]}
    context["previous_rounds"] = memory["rounds"][-6:]
    context["previous"] = memory["rounds"][-1] if memory["rounds"] else {}
    context["rejections"] = _rejections(memory["rounds"])
    context["stage_id"] = getattr(rnd, "stage_id", "") or ""
    previous_stage = memory["rounds"][-1].get("stage_id") if memory["rounds"] else None
    context["stage_changed"] = bool(context["stage_id"]) and previous_stage not in (None, context["stage_id"])
    plan, call_id = contest_llm.decide(rnd, context)
    # 平台要求始终基于父检查点续训：无论模型给出什么动作，本轮都执行真实续训。
    plan["action"] = "train"
    if context["stage_changed"] and plan["epochs"] < 2:
        plan["epochs"] = 2
    _log("[%s] round=%d stage=%r changed=%s rejections=%d plan=%s"
         % (key, rnd.round_index, context["stage_id"], context["stage_changed"],
            context["rejections"], json.dumps(plan, ensure_ascii=False)))

    from baselines.train import train_run  # 延迟导入，describe 阶段不加载 numpy

    trained = work / "trained"
    # 强制续训：父检查点缺失就立刻失败，绝不从头建模型。
    if not parent.is_file():
        raise RuntimeError("父检查点不存在，拒绝从头训练：%s" % parent)
    report = train_run(
        config, train_path, validation_path, trained,
        resume=parent,
        epochs=plan["epochs"],
        learning_rate=round(baseline_lr * plan["learning_rate_scale"], 12),
        seed=None,
        stage_id=context["stage_id"] or ("round-%d" % rnd.round_index),
        label_provenance=_provenance(work) if key == "B" else None,
        primary_metric=context["metric"],
        deadline=_deadline(rnd, round_started),
        efficiency_meta=filter_meta,
    )
    _log("[timing] train=%.2fs report=%.2fs epochs=%s strategy=%s"
         % (time.monotonic() - round_started, float(report.get("training_seconds") or 0.0),
            report.get("epochs_completed"),
            (report.get("efficiency") or {}).get("strategy")))
    tolerance = 0.005 if context["stage_changed"] else 0.0
    gate = _evaluate(context["metric"], report["validation_before"], report["validation_after"],
                     tolerance=tolerance, direction=context.get("metric_direction", ""))
    gate.update(_guard(context["metric"], report["validation_before"], report["validation_after"]))
    gate["decision"] = "train"
    accepted = bool(gate.get("improved")) and not gate.get("vetoed")

    # 候选永远是真实训练产物；是否采用由 finish 的 selected_artifact_id 决定。
    candidate = work / "candidate"
    candidate.mkdir(parents=True)
    shutil.copyfile(trained / "checkpoint.npz", candidate / "checkpoint.npz")
    shutil.copyfile(trained / "history.jsonl", candidate / "history.jsonl")
    gate["selected"] = "trained" if accepted else "parent"

    # 自写训练报告：只写可核验的事实，且必须与候选目录里实际的 checkpoint 一致。
    candidate_sha = _sha256(candidate / "checkpoint.npz")
    parent_sha = _sha256(parent)
    probe = {
        "round_index": rnd.round_index,
        "stage_id": context["stage_id"],
        "stage_changed": bool(context["stage_changed"]),
        "primary_metric": context["metric"],
        "validation_before": report["validation_before"],
        "validation_after": report["validation_after"],
        "gap_to_parent": _gap(context["metric"], report["validation_before"], report["validation_after"]),
        "selected_epoch": report.get("selected_epoch"),
        "epochs_completed": report.get("epochs_completed"),
        "epoch_stop_reason": report.get("epoch_stop_reason"),
        "validation_by_epoch": report.get("validation_by_epoch"),
        "efficiency": report.get("efficiency"),
        "rejections_before": context["rejections"],
        "accepted": accepted,
    }
    (candidate / "probe.json").write_text(
        json.dumps(probe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    training_report = {
        "format": "contest.training-report.v1",
        "dataset": key,
        "round_index": rnd.round_index,
        "stage_id": context["stage_id"],
        "resumed": True,
        "trained_from_parent": True,
        "parent_artifact_id": rnd.parent.artifact_id,
        "parent_checkpoint_sha256": parent_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "selected_checkpoint_sha256": candidate_sha if accepted else parent_sha,
        "kept_parent": not accepted,
        "selected": gate["selected"],
        "epochs_requested": plan["epochs"],
        "epochs_ran": int(report.get("epochs_completed") or 0),
        "initial_steps": int(report.get("initial_steps") or 0),
        "final_steps": int(report.get("final_steps") or 0),
        "primary_metric": context["metric"],
        "metric_direction": context["metric_direction"],
        "validation_before": report["validation_before"],
        "validation_after": report["validation_after"],
        "improved": bool(gate.get("improved")),
        "vetoed": bool(gate.get("vetoed")),
        "secondary": gate.get("secondary"),
    }
    (candidate / "training.json").write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    (candidate / "decision.json").write_text(json.dumps({
        "dataset": key, "round_index": rnd.round_index, "plan": plan, "gate": gate,
        "model_call_id": call_id,
        "parent_artifact_id": rnd.parent.artifact_id,
        "resumed": True,
        "validation_before": (report or {}).get("validation_before"),
        "validation_after": (report or {}).get("validation_after"),
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    state = work / "state"
    state.mkdir()
    rounds = list(memory["rounds"]) + [{
        "dataset": key, "round_index": rnd.round_index, "stage_id": context["stage_id"],
        "action": plan["action"], "epochs": plan["epochs"],
        "learning_rate_scale": plan["learning_rate_scale"],
        "metric": gate["metric"], "metric_value": gate.get("after"),
        "before": gate.get("before"), "after": gate.get("after"),
        "improved": gate.get("improved"), "accepted": accepted,
        "vetoed": bool(gate.get("vetoed")), "selected": gate.get("selected"),
        "selected_epoch": report.get("selected_epoch"),
        "hypothesis": plan["hypothesis"][:160],
    }]
    (state / "memory.json").write_text(json.dumps({"rounds": rounds[-20:]}, ensure_ascii=False,
                                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rnd.save_state(state)

    seed = int(report.get("seed", config_body["seed"]))
    configuration = {
        "dataset": key, "task": context["task"], "metric": context["metric"],
        "epochs": plan["epochs"], "learning_rate_scale": plan["learning_rate_scale"],
        "learning_rate": round(baseline_lr * plan["learning_rate_scale"], 12),
        "stage_id": context["stage_id"] or ("round-%d" % rnd.round_index), "seed": seed,
        "action": "train", "decision_source": plan["source"], "selected": gate.get("selected"),
        "selected_epoch": report.get("selected_epoch"),
        "model_call_id": call_id, "gate": gate,
        "history_ids": [item["id"] for item in history if isinstance(item.get("id"), str)],
    }
    reason = "%s；%s 前=%s 后=%s（保存第 %s 个 epoch），%s。" % (
        plan["reason"], gate["metric"], gate.get("before"), gate.get("after"),
        report.get("selected_epoch"),
        "采用本轮训练产物" if accepted else "本轮候选未超过父检查点，选定父产物以保持单调不降")
    configuration["reason"] = _clip(reason, 1200)
    submitted = rnd.submit_candidate(
        candidate,
        hypothesis=plan["hypothesis"],
        change_summary=plan["change_summary"],
        configuration=configuration,
        seed=seed,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    # 候选已经作为真实训练产物提交并被平台测量；这里只决定本轮"选中"哪一个。
    selected_id = submitted["artifact_id"] if accepted else rnd.parent.artifact_id
    elapsed = time.monotonic() - round_started
    _log("[submit] artifact=%s accepted=%s selected=%s total=%.2fs extra=%s"
         % (submitted.get("artifact_id"), accepted, selected_id, elapsed,
            json.dumps({k: v for k, v in submitted.items() if k != "artifact_id"}, ensure_ascii=False)[:600]))
    rnd.finish(
        selected_artifact_id=selected_id,
        hypothesis=plan["hypothesis"],
        change_summary=plan["change_summary"],
        reason=_clip(reason, 3800),
        configuration=configuration,
        seed=seed,
        history_ids=configuration["history_ids"],
        model_call_ids=list(rnd.call_ids),
    )


_MODEL = {}


def _model(checkpoint: Path):
    key = str(checkpoint)
    model = _MODEL.get(key)
    if model is None:
        from baselines.model import Model
        model = Model.load(checkpoint)
        _MODEL[key] = model
    return model


def predict(sample, model_dir=None):
    del model_dir
    model = _model(_checkpoint_path())
    import numpy as np
    from baselines.data import flatten

    feature_paths, values = flatten(sample.inputs)
    if feature_paths != model.meta["paths"]:
        values = contest_data.vector(sample.inputs, model.meta["paths"])
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if model.meta["task"] == "regression":
        return {"value": float(max(0.0, output))}
    scores = [float(min(1.0, max(0.0, value))) for value in output]
    top = int(np.argmax(scores))
    if scores.count(scores[top]) > 1:
        scores[top] = min(1.0, scores[top] + 1e-6)
    return [{"label": label, "score": score} for label, score in zip(model.meta["classes"], scores)]


def main(argv):  # 便于本地直接跑 train/predict
    from contest_harness.cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        raise
