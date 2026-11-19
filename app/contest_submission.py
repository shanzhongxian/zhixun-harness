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


def _log(message: str) -> None:
    print(str(message), file=sys.stderr, flush=True)


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


def _materialize(rnd, work: Path):
    train_path = work / "train.jsonl"
    validation_path = work / "validation.jsonl"
    train_rows = contest_data.write_jsonl(rnd.data.training_rows(), train_path)
    validation_rows = contest_data.write_jsonl(rnd.data.validation_rows(), validation_path)
    if not train_rows or not validation_rows:
        raise RuntimeError("公开数据为空：train=%d validation=%d" % (train_rows, validation_rows))
    return train_path, validation_path, train_rows, validation_rows


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


def _evaluate(metric: str, before, after) -> dict:
    direction = _METRIC_DIRECTION.get(metric, "higher")
    before = dict(before or {})
    after = dict(after or {})
    value_before = before.get(metric)
    value_after = after.get(metric)
    improved = value_after is not None and value_before is not None and (
        value_after > value_before if direction == "higher" else value_after < value_before)
    return {"metric": metric, "direction": direction, "before": value_before, "after": value_after,
            "improved": bool(improved)}


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
    key = contest_data.dataset_key(rnd.dataset_key)
    work = paths.workspace() / ("round-%02d-%s" % (rnd.round_index, key))
    index = 0
    while work.exists():
        index += 1
        work = paths.workspace() / ("round-%02d-%s-%d" % (rnd.round_index, key, index))
    work.mkdir(parents=True)
    memory = _load_memory(rnd, work)
    train_path, validation_path, train_rows, validation_rows = _materialize(rnd, work)
    parent = _parent_checkpoint(rnd)
    history = rnd.history()
    config = _config_path(key)
    config_body = json.loads(config.read_text(encoding="utf-8"))
    baseline_lr = float(config_body["learning_rate"])

    context = _summarize(rnd, key, history, train_rows, validation_rows)
    context["parent"] = {"artifact_id": rnd.parent.artifact_id, "checkpoint": parent.name,
                         "baseline_learning_rate": baseline_lr, "baseline_epochs": config_body["epochs"]}
    context["previous_rounds"] = memory["rounds"][-6:]
    plan, call_id = contest_llm.decide(rnd, context)
    _log("[%s] round=%d plan=%s" % (key, rnd.round_index, json.dumps(plan, ensure_ascii=False)))

    from baselines.train import train_run  # 延迟导入，describe 阶段不加载 numpy

    trained = work / "trained"
    report = None
    if plan["action"] == "no_op":
        gate = {"metric": context["metric"], "direction": context["metric_direction"],
                "before": None, "after": None, "improved": False, "decision": "no_op"}
    else:
        report = train_run(
            config, train_path, validation_path, trained,
            resume=parent,
            epochs=plan["epochs"],
            learning_rate=round(baseline_lr * plan["learning_rate_scale"], 12),
            seed=None,
            stage_id="round-%d" % rnd.round_index,
            label_provenance=_provenance(work) if key == "B" else None,
        )
        gate = _evaluate(context["metric"], report["validation_before"], report["validation_after"])
        gate["decision"] = "train"

    # 训练没变好就交回父代权重：提交链只允许单调不降。
    keep_trained = bool(report) and gate.get("improved") is True
    candidate = work / "candidate"
    candidate.mkdir(parents=True)
    shutil.copyfile(trained / "checkpoint.npz" if keep_trained else parent, candidate / "checkpoint.npz")
    if report is not None:
        for name in ("training.json", "history.jsonl"):
            shutil.copyfile(trained / name, candidate / name)
    gate["selected"] = "trained" if keep_trained else "parent"

    (candidate / "decision.json").write_text(json.dumps({
        "dataset": key, "round_index": rnd.round_index, "plan": plan, "gate": gate,
        "model_call_id": call_id,
        "validation_before": (report or {}).get("validation_before"),
        "validation_after": (report or {}).get("validation_after"),
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    state = work / "state"
    state.mkdir()
    rounds = list(memory["rounds"]) + [{
        "dataset": key, "round_index": rnd.round_index, "action": plan["action"],
        "epochs": plan["epochs"], "learning_rate_scale": plan["learning_rate_scale"],
        "metric": gate["metric"], "before": gate.get("before"), "after": gate.get("after"),
        "improved": gate.get("improved"), "selected": gate.get("selected"),
        "hypothesis": plan["hypothesis"][:160],
    }]
    (state / "memory.json").write_text(json.dumps({"rounds": rounds[-20:]}, ensure_ascii=False,
                                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rnd.save_state(state)

    seed = int((report or {}).get("seed", config_body["seed"]))
    configuration = {
        "dataset": key, "task": context["task"], "metric": context["metric"],
        "epochs": plan["epochs"], "learning_rate_scale": plan["learning_rate_scale"],
        "learning_rate": round(baseline_lr * plan["learning_rate_scale"], 12),
        "stage_id": "round-%d" % rnd.round_index, "seed": seed,
        "action": plan["action"], "decision_source": plan["source"], "selected": gate.get("selected"),
        "model_call_id": call_id, "gate": gate, "history_ids": [item["id"] for item in history if isinstance(item.get("id"), str)],
    }
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
    reason = "%s；%s 前=%s 后=%s，本轮提交%s。" % (
        plan["reason"], gate["metric"], gate.get("before"), gate.get("after"),
        "训练产物" if keep_trained else "父代权重（训练未变好，保持单调不降）")
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=plan["hypothesis"],
        change_summary=plan["change_summary"],
        reason=reason[:3800],
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
