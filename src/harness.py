"""智训引擎 harness：诊断 → 决策 → 训练 → 门禁 → 版本管理。

复用 project.py 的续训入口与评测实现，包一层"训练师决策 + 门禁 + 版本指针"。

用法示例：
    python harness.py --profile A --strategy rule          # 规则决策，续训 A
    python harness.py --profile all --strategy mock        # 模拟训练师，跑三项
    python harness.py --profile A --strategy qwen          # 真 Qwen，失败自动降级规则

三条关键路径（自测见 selftest.py）：
    候选通过(accepted) / 候选退化(rejected) / 非法决策降级(fallback)。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent          # src/
ROOT = SRC.parent                              # 包根目录（智训引擎/）
sys.path.insert(0, str(SRC))

# 复用 project.py（含续训入口 train_profile、评测 evaluate、数据加载等）。
import project

from strategy.schemas import (Decision, validate_decision, PRIMARY_METRIC, DIRECTION,
                              GROUP_KEYS, LR_SCALE_DEFAULTS)
from strategy.observability import compute_diagnostics, load_meta, extract_group
from strategy.deciders import rule_decide, mock_decide, qwen_decide
from strategy.replay import build_split, masks_for
from runtime.swanlab_log import SwanLabReporter

# 版本与决策日志目录；可用 HARNESS_RUNS_DIR 覆盖（自测用临时目录）。
RUNS = Path(os.environ.get("HARNESS_RUNS_DIR", str(ROOT / "runs")))
CURRENT_JSON = RUNS / "current.json"
DECISIONS_JSONL = RUNS / "decisions.jsonl"


def _resolve_stored(value):
    """把落盘路径解析成本机可用路径；兼容旧的 E:\\项目\\1\\... 绝对路径。"""
    if not value:
        return None
    p = Path(value)
    if p.exists():
        return p
    parts = p.parts
    for anchor in ("runs", "packages", ".selftest"):
        if anchor in parts:
            candidate = ROOT.joinpath(*parts[parts.index(anchor):])
            if candidate.exists():
                return candidate
    return p


def _load_current() -> dict:
    if CURRENT_JSON.exists():
        raw = json.loads(CURRENT_JSON.read_text(encoding="utf-8"))
        for entry in raw.values():
            for key in ("checkpoint", "previous"):
                if entry.get(key):
                    entry[key] = str(_resolve_stored(entry[key]))
        return raw
    current = {}
    for profile in project.PACKAGES:
        current[profile] = {
            "checkpoint": str(project.PACKAGES[profile] / "checkpoint.npz"),
            "previous": None,
            "best_metric": None,
            "rounds": 0,
        }
    return current


def _save_current(current: dict) -> None:
    """落盘时把仓库内路径转成相对路径，换机器/进镜像都不会失效。"""
    RUNS.mkdir(parents=True, exist_ok=True)
    portable = {}
    for profile, entry in current.items():
        copy = dict(entry)
        for key in ("checkpoint", "previous"):
            if copy.get(key):
                copy[key] = project.to_store_path(copy[key])
        portable[profile] = copy
    CURRENT_JSON.write_text(json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _log_decision(record: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    with DECISIONS_JSONL.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _config(profile) -> dict:
    package = project.PACKAGES[profile]
    return json.loads((package / "baselines" / "configs" / f"{profile}.json").read_text(encoding="utf-8"))


def _metric_at(profile, checkpoint) -> dict:
    """在公开验证集（门禁集）上评测一个检查点，返回完整指标。"""
    config = _config(profile)
    model = project.Model.load(checkpoint)
    package = project.PACKAGES[profile]
    validation = project.read_dataset(
        package / "public" / "validation.jsonl", config["task"], config.get("classes", []), model.meta["paths"]
    )
    return project.evaluate(profile, model, validation, nasa=(profile == "C"))


def _primary(profile, model, x, y) -> float:
    """在给定子集 (x, y) 上算主指标（不附带 NASA 分）。"""
    return project.evaluate(profile, model, {"x": x, "y": y}, nasa=False)[PRIMARY_METRIC[profile]]


def _decide(strategy, diagnostics, profile, *, lr_scale, epochs, patience, model_id, attempts=0, rejections=0) -> Decision:
    if strategy == "qwen":
        decision = qwen_decide(diagnostics, profile, model_id=model_id, attempts=attempts, rejections=rejections)
        if decision is None:
            decision = rule_decide(diagnostics, profile, lr_scale=lr_scale, epochs=epochs, patience=patience)
            decision.source = "rule"
            decision.hypothesis = "（Qwen 不可用，降级到规则）" + decision.hypothesis
        else:
            decision = decision
    elif strategy == "mock":
        decision = mock_decide(diagnostics, profile, lr_scale=lr_scale, epochs=epochs, patience=patience)
    else:
        decision = rule_decide(diagnostics, profile, lr_scale=lr_scale, epochs=epochs, patience=patience)

    clean, errors = validate_decision(decision.to_dict() if isinstance(decision, Decision) else decision, profile)
    if errors:
        clean = Decision("no_op", "决策非法，降级为 no_op：" + "; ".join(errors),
                         evidence_ids=[], params={}, source="fallback")
    return clean


def run_round(profile, strategy, *, lr_scale, epochs, patience, batch_size, output_dir,
              min_improvement, model_id, extra_train=None, reporter=None, use_policy=True) -> dict:
    current = _load_current()
    entry = current[profile]
    config = _config(profile)
    task = config["task"]
    classes = config.get("classes", [])
    package = project.PACKAGES[profile]
    primary = PRIMARY_METRIC[profile]
    direction = DIRECTION[profile]

    policy = None
    if use_policy:
        try:
            from policy.store import PolicyStore
            policy = PolicyStore(ROOT).champion()
        except Exception:
            policy = None
    policy_thresholds = policy.thresholds_for(profile) if policy else None

    model = project.Model.load(entry["checkpoint"])
    train = project.read_dataset(package / "public" / "train.jsonl", task, classes, model.meta["paths"])
    validation = project.read_dataset(package / "public" / "validation.jsonl", task, classes, model.meta["paths"])
    validation_meta = load_meta(package / "public" / "validation.jsonl")
    groups = [extract_group(m, profile) for m in validation_meta]
    split = build_split(groups)
    gate_mask, replay_mask = masks_for(groups, split)

    def evaluate(data, nasa=False):
        return project.evaluate(profile, model, data, nasa=nasa)

    diagnostics = compute_diagnostics(profile, model, train, validation, evaluate, validation_meta,
                                      thresholds=policy_thresholds)
    if extra_train is not None:
        batch_metrics = project.evaluate(profile, model, extra_train, nasa=(profile == "C"))
        diagnostics["batch_metrics"] = batch_metrics
        diagnostics["batch_drift_flag"] = bool(
            direction * (batch_metrics[primary] - diagnostics["validation_metrics"][primary]) < -min_improvement
        )
    attempts = entry.get("attempts", 0)
    consecutive_rejections = entry.get("consecutive_rejections", 0)
    decision = _decide(strategy, diagnostics, profile, lr_scale=lr_scale, epochs=epochs,
                       patience=patience, model_id=model_id,
                       attempts=attempts, rejections=consecutive_rejections)

    # T23 必须尝试：即使 Qwen/规则判 no_op，也用小步长试一次（可改不成功，但不能不尝试）
    if decision.action == "no_op":
        decision = Decision(
            "train",
            "必须尝试：即使当前判断无需改变，也用小步长试一次（可改不成功但不能不尝试）",
            evidence_ids=[], params={"lr_scale": 0.1, "epochs": 2, "patience": 1},
            source=decision.source,
        )

    if reporter is None:
        reporter = SwanLabReporter(ROOT)
    reporter.start(profile, entry["rounds"] + 1, config={
        "strategy": strategy, "action": decision.action,
        "hypothesis": decision.hypothesis, "params": decision.params, "model_id": model_id,
        "policy_id": policy.id if policy else None,
        "policy_parent": policy.parent if policy else None,
    })

    current_full = diagnostics["validation_metrics"][primary]
    current_replay = _primary(profile, model, validation["x"][replay_mask], validation["y"][replay_mask])
    record = {
        "profile": profile,
        "strategy": strategy,
        "round": entry["rounds"] + 1,
        "attempts_before": attempts,
        "consecutive_rejections_before": consecutive_rejections,
        "action": decision.action,
        "hypothesis": decision.hypothesis,
        "evidence_ids": decision.evidence_ids,
        "params": decision.params,
        "decision_source": decision.source,
        "diagnostics": diagnostics,
        "replay_split": {"gate": [list(g) for g in split["gate"]], "replay": [list(g) for g in split["replay"]]},
        "current_checkpoint": entry["checkpoint"],
        "new_checkpoint": None,
        "extra_train": {"rows": len(extra_train["ids"]), "sha256": extra_train["sha256"]} if extra_train is not None else None,
        "policy_id": policy.id if policy else None,
        "policy_parent": policy.parent if policy else None,
        "result": None,
    }

    if decision.action == "no_op":
        record["result"] = "no_op"
    elif decision.action == "rollback":
        if entry.get("previous"):
            entry["checkpoint"], entry["previous"] = entry["previous"], None
            record["result"] = "rolled_back"
            record["new_checkpoint"] = entry["checkpoint"]
        else:
            record["result"] = "rollback_no_previous"
    elif decision.action == "train":
        params = decision.params
        psearch = policy.search_for(profile) if policy else {}
        if psearch:
            from policy.bounds import clamp_params
            params, clamp_notes = clamp_params(params, psearch)
            record["clamp_notes"] = clamp_notes
        # 优先级：决策参数 > 冠军策略 > 调用方默认值（决策参数已按策略边界夹取）
        run_lr = params.get("lr_scale") or psearch.get("lr_scale") or lr_scale
        run_epochs = params.get("epochs") or psearch.get("epochs") or epochs
        run_batch = params.get("batch_size") or batch_size
        run_patience = params.get("patience") or psearch.get("patience") or patience
        run_wd = params["weight_decay"] if "weight_decay" in params else psearch.get("weight_decay", 0.0)
        record["applied_params"] = {"lr_scale": run_lr, "epochs": run_epochs,
                                    "patience": run_patience, "weight_decay": run_wd,
                                    "source": "decision" if decision.params else ("policy" if psearch else "cli"),
                                    "policy_id": policy.id if policy else None}
        run_dropout = params.get("dropout")
        run_loss = params.get("loss")
        output = ROOT / output_dir / f"round-{entry['rounds'] + 1}" / profile
        report = project.train_profile(
            profile, output, epochs=run_epochs, lr_scale=run_lr, batch_size=run_batch,
            patience=run_patience, resume=Path(entry["checkpoint"]), weight_decay=run_wd,
            extra_train=extra_train, dropout=run_dropout, loss=run_loss,
        )
        candidate = output / "checkpoint.npz"
        candidate_metric = _metric_at(profile, candidate)
        cand_model = project.Model.load(candidate)
        cand_replay = _primary(profile, cand_model, validation["x"][replay_mask], validation["y"][replay_mask])
        improved = direction * (candidate_metric[primary] - current_full) > min_improvement
        replay_degraded = direction * (cand_replay - current_replay) < -min_improvement
        accepted = improved and not replay_degraded
        record["result"] = "accepted" if accepted else "rejected"
        record["new_checkpoint"] = str(candidate)
        record["candidate_report"] = report
        record["candidate_metric"] = candidate_metric
        record["replay_metric"] = {"current": current_replay, "candidate": cand_replay, "degraded": replay_degraded}
        if accepted:
            entry["previous"] = entry["checkpoint"]
            entry["checkpoint"] = str(candidate)
            entry["best_metric"] = candidate_metric[primary]
    else:
        record["result"] = "unknown_action"

    if decision.action == "train":
        entry["attempts"] = attempts + 1
        if record["result"] == "rejected":
            entry["consecutive_rejections"] = consecutive_rejections + 1
        elif record["result"] == "accepted":
            entry["consecutive_rejections"] = 0
    entry["rounds"] += 1
    current[profile] = entry
    _save_current(current)
    _log_decision(record)

    metrics = {
        "primary": diagnostics["validation_metrics"][primary],
        "train_loss": diagnostics["train_loss"],
        "validation_loss": diagnostics["validation_loss"],
        "gap": diagnostics["gap"],
        "n_groups": diagnostics["n_groups"],
        "overfit": int(diagnostics["overfit_flag"]),
        "drift": int(diagnostics["drift_flag"]),
        "accepted": 1 if record["result"] == "accepted" else 0,
    }
    if record.get("candidate_metric"):
        metrics["candidate_primary"] = record["candidate_metric"][primary]
    reporter.log(metrics, step=record["round"])
    reporter.finish()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("A", "B", "C", "all"), default="A")
    parser.add_argument("--strategy", choices=("rule", "mock", "qwen"), default="rule")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr-scale", type=float, default=None, help="缺省按任务 A=0.3 B=0.5 C=0.3")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.0, help="定时监控间隔（秒）；0=跑完即停")
    parser.add_argument("--min-improvement", type=float, default=1e-6)
    parser.add_argument("--output-dir", default="runs/rounds")
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--no-policy", action="store_true", help="忽略冠军策略，只用命令行/决策参数")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200 or args.patience < 1:
        parser.error("--epochs 必须在 1..200，--patience 必须大于 0")

    profiles = list(project.PACKAGES) if args.profile == "all" else [args.profile]
    for cycle in range(args.rounds):
        for profile in profiles:
            lr_scale = args.lr_scale if args.lr_scale is not None else LR_SCALE_DEFAULTS[profile]
            record = run_round(
                profile, args.strategy, lr_scale=lr_scale, epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, output_dir=args.output_dir,
                min_improvement=args.min_improvement, model_id=args.model_id,
                use_policy=not args.no_policy,
            )
            primary = record["diagnostics"]["primary_metric"]
            print(f"[{profile}] round={record['round']} action={record['action']} "
                  f"source={record['decision_source']} result={record['result']} "
                  f"{primary}={record['diagnostics']['validation_metrics'][primary]:.6f}", flush=True)
            print(f"    hypothesis: {record['hypothesis']}", flush=True)
        if args.interval > 0 and cycle < args.rounds - 1:
            print(f"  ... 定时监控：等待 {args.interval:.0f}s 后进入下一轮 ...", flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
