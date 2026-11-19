"""用随包检查点续训 A 气体识别、B 轴承 RUL、C 发动机 RUL。

相对原版修正的思路问题：
1. 早停与“最佳候选”以父检查点的验证指标为基准，而不是以续训后的第一个 epoch 为基准。
   续训若始终没有超过父检查点，默认保留父检查点作为本轮候选，避免提交一个退化的模型；
   传 --keep-best-continued 可改回“总是保存续训内最佳候选”。
2. 训练集指标只在开始与结束各算一次，不再每个 epoch 做一次全量前向；C 的 NASA 分数只在验证集上算。
3. 学习率倍率按任务提供默认值（B 数据量多、RMSE 仍高，可用稍大倍率），仍可显式覆盖。
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


SRC = Path(__file__).resolve().parent          # src/
ROOT = SRC.parent                              # 包根目录（智训引擎/）

# 包名带日期（cpu-a-guide-20261110-v1），平台换包就失效，因此按 baseline-info.json
# 的 dataset_key 自动发现；找不到时退回已知目录名，保证老环境仍能跑。
_LEGACY_PACKAGES = {
    "A": "cpu-a-guide-20261110-v1",
    "B": "cpu-b-rul-20261110-v1",
    "C": "cpu-c-guide-20261110-v1",
}


def discover_packages(root: Path | None = None) -> dict:
    """扫描 packages/*/baseline-info.json，按 dataset_key 返回 {A/B/C: 目录}。"""
    root = root or ROOT
    found: dict = {}
    for info in sorted((root / "packages").glob("*/baseline-info.json")):
        try:
            key = str(json.loads(info.read_text(encoding="utf-8")).get("dataset_key", "")).strip().upper()
        except Exception:
            continue
        if key in ("A", "B", "C"):
            found.setdefault(key, info.parent)
    for key, name in _LEGACY_PACKAGES.items():
        found.setdefault(key, root / "packages" / name)
    return found


PACKAGES = discover_packages()


def to_store_path(path) -> str:
    """落盘用：仓库内路径存相对（POSIX 风格），仓库外保持绝对。"""
    if not path:
        return ""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def from_store_path(value) -> Path:
    """读取用：相对路径按仓库根解析，绝对路径原样返回。"""
    if not value:
        return Path("")
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p)

# 三个训练包的模型和数据加载代码相同。轴承包包含新版 RUL 配置。
sys.path.insert(0, str(PACKAGES["B"]))
from baselines.model import Model, checkpoint_digest
from baselines.data import read_dataset
from baselines.train import metrics
import numpy as np

# 各任务续训的默认学习率倍率（相对随包默认学习率）。
LR_SCALE_DEFAULTS = {"A": 0.3, "B": 0.5, "C": 0.3}


def evaluate(profile, model, data, *, nasa=False):
    result = metrics(model, data)
    if profile == "C" and nasa:
        predicted = model.predict(data["x"])
        error = predicted - data["y"]
        # NASA C-MAPSS 非对称评分。相同绝对误差下，高估寿命惩罚更大。
        exponent = np.where(error > 0, error / 10, -error / 13)
        result["nasa_score"] = float(np.expm1(exponent).sum())
        result["overestimate_rate"] = float(np.mean(error > 0))
    return result


def _merge_train(base, extra):
    """把批次数据合并到公开训练集（同一 schema，按行拼接）。"""
    return {
        "ids": base["ids"] + extra["ids"],
        "x": np.concatenate([base["x"], extra["x"]]),
        "y": np.concatenate([base["y"], extra["y"]]),
        "paths": base["paths"],
        "sha256": base["sha256"],
    }


def train_profile(profile, output, *, epochs, lr_scale, batch_size, patience, resume=None, keep_best_continued=False, weight_decay=0.0, extra_train=None, dropout=None, loss=None):
    """从已有权重续训。每轮只用公开训练集更新，公开验证集用于早停。"""
    if not 0 < lr_scale <= 1:
        raise ValueError("--lr-scale 必须大于 0 且不超过 1")
    package = PACKAGES[profile]
    config = json.loads((package / "baselines" / "configs" / f"{profile}.json").read_text(encoding="utf-8"))
    parent = resume if resume is not None else package / "checkpoint.npz"
    model = Model.load(parent)
    task = config["task"]
    classes = config.get("classes", [])
    if (model.meta.get("profile"), model.meta["task"], model.meta["classes"], model.meta["hidden"]) != (
        profile, task, classes, config["hidden"]
    ):
        raise ValueError(f"{profile}：检查点与训练包的任务、类别或网络结构不匹配")
    if profile == "B" and model.meta.get("target_unit") != "minutes":
        raise ValueError("B：轴承检查点必须使用分钟作为回归单位")

    train = read_dataset(package / "public" / "train.jsonl", task, classes, model.meta["paths"])
    validation = read_dataset(package / "public" / "validation.jsonl", task, classes, model.meta["paths"])
    if extra_train is not None:
        if extra_train.get("paths") != model.meta["paths"]:
            raise ValueError(f"{profile}：批次特征 schema 与检查点不匹配")
        train = _merge_train(train, extra_train)
    if set(train["ids"]).intersection(validation["ids"]):
        raise ValueError(f"{profile}：训练集和验证集的样本 ID 重叠")

    initial_steps = model.meta["steps"]
    initial_epochs = model.meta["epochs"]
    baseline = evaluate(profile, model, validation, nasa=(profile == "C"))
    train_before = evaluate(profile, model, train)
    primary = "macro_f1" if profile == "A" else "rmse"
    direction = 1 if profile == "A" else -1
    model.meta["learning_rate"] = config["learning_rate"] * lr_scale
    model.meta["weight_decay"] = float(weight_decay)
    if dropout is not None:
        model.meta["dropout"] = float(dropout)
    if loss is not None:
        model.meta["loss"] = loss
    if batch_size is not None:
        model.meta["batch_size"] = batch_size
    model.validate()

    history = []
    best_model = None                 # None 表示父检查点仍是当前最佳
    best_validation = baseline        # 早停基准从父检查点开始，而不是第一个续训 epoch
    best_epoch = None
    best_continued_model = None       # 续训内最佳模型（仅 keep_best_continued 时保存权重）
    best_continued_validation = None
    best_continued_epoch = None
    epochs_without_improvement = 0
    for _ in range(epochs):
        loss = model.fit(train["x"], train["y"], 1)[0]
        validation_metrics = evaluate(profile, model, validation, nasa=(profile == "C"))
        history.append({
            "epoch": model.meta["epochs"],
            "step": model.meta["steps"],
            "loss": loss,
            "validation": validation_metrics,
        })
        if best_continued_validation is None or direction * (validation_metrics[primary] - best_continued_validation[primary]) > 0:
            best_continued_validation = validation_metrics
            best_continued_epoch = model.meta["epochs"]
            if keep_best_continued:
                best_continued_model = deepcopy(model)
        if direction * (validation_metrics[primary] - best_validation[primary]) > 0:
            best_model = deepcopy(model)
            best_validation = validation_metrics
            best_epoch = model.meta["epochs"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        print(f"  [{profile}] 续训轮 {len(history)}/{epochs}: loss={loss:.6f}, "
              f"{primary}={validation_metrics[primary]:.6f}", flush=True)

    improved_primary = direction * (best_validation[primary] - baseline[primary]) > 0
    if best_model is not None:
        model_to_save, validation_after, saved_epoch, retained_parent = best_model, best_validation, best_epoch, False
    elif keep_best_continued and best_continued_model is not None:
        model_to_save, validation_after, saved_epoch, retained_parent = best_continued_model, best_continued_validation, best_continued_epoch, False
    else:
        model_to_save, validation_after, saved_epoch, retained_parent = None, baseline, None, True

    # 显式判空，静态检查（Pylance）才能把 model_to_save 收窄为 Model。
    if model_to_save is not None:
        final_steps = model_to_save.meta["steps"]
        final_epochs = model_to_save.meta["epochs"]
        train_after = evaluate(profile, model_to_save, train)
    else:
        final_steps = initial_steps
        final_epochs = initial_epochs
        train_after = train_before

    # 保存本轮候选：若续训始终没超过父检查点且未要求保留续训候选，则沿用父检查点。
    report = {
        "profile": profile,
        "task": task,
        "parameter_count": model.meta["parameter_count"],
        "seed": model.meta["seed"],
        "primary_metric": primary,
        "train_rows": len(train["ids"]),
        "validation_rows": len(validation["ids"]),
        "train_sha256": train["sha256"],
        "validation_sha256": validation["sha256"],
        "extra_train_rows": len(extra_train["ids"]) if extra_train is not None else 0,
        "extra_train_sha256": extra_train.get("sha256") if extra_train is not None else None,
        "parent_checkpoint_sha256": checkpoint_digest(parent),
        "learning_rate": model.meta["learning_rate"],
        "batch_size": model.meta["batch_size"],
        "weight_decay": model.meta.get("weight_decay", 0.0),
        "dropout": model.meta.get("dropout", 0.0),
        "loss": model.meta.get("loss", "mse"),
        "requested_epochs": epochs,
        "ran_epochs": len(history),
        "patience": patience,
        "initial_steps": initial_steps,
        "initial_epochs": initial_epochs,
        "best_epoch": saved_epoch,
        "retained_parent": retained_parent,
        "final_steps": final_steps,
        "final_epochs": final_epochs,
        "validation_before": baseline,
        "validation_after": validation_after,
        "improved_primary_metric": improved_primary,
        "train_before": train_before,
        "train_after": train_after,
    }

    if output.exists() or output.is_symlink():
        raise ValueError(f"输出目录已存在，请为新实验换一个目录：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        checkpoint = staging / "checkpoint.npz"
        if model_to_save is not None:
            model_to_save.save(checkpoint)
        else:
            shutil.copyfile(parent, checkpoint)
        report["checkpoint_sha256"] = checkpoint_digest(checkpoint)
        (staging / "training.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        with (staging / "history.jsonl").open("w", encoding="utf-8") as stream:
            for record in history:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=5, help="最多续训多少轮")
    parser.add_argument("--lr-scale", type=float, default=None,
                        help="相对随包默认学习率的倍率；缺省为 A=0.3、B=0.5、C=0.3")
    parser.add_argument("--batch-size", type=int, help="默认沿用检查点的批大小")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 权重衰减（0 到 1）")
    parser.add_argument("--patience", type=int, default=2, help="验证主指标连续几轮不超过父检查点就停止")
    parser.add_argument("--keep-best-continued", action="store_true",
                        help="即使没超过父检查点也保存续训内最佳候选（默认保留父检查点）")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/round-1"), help="相对于脚本目录")
    parser.add_argument("--resume", type=Path, help="上一轮的 .npz 路径，相对于脚本目录，仅用于单任务")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10000 or args.patience < 1:
        parser.error("--epochs 必须在 1 到 10000 之间，--patience 必须大于 0")
    if args.lr_scale is not None and not 0 < args.lr_scale <= 1:
        parser.error("--lr-scale 必须大于 0 且不超过 1")
    if args.batch_size is not None and not 1 <= args.batch_size <= 65536:
        parser.error("--batch-size 必须在 1 到 65536 之间")
    if not 0 <= args.weight_decay <= 1:
        parser.error("--weight-decay 必须在 0 到 1 之间")
    if args.resume is not None and args.profile == "all":
        parser.error("--resume 需要指定单个 --profile")

    profiles = list(PACKAGES) if args.profile == "all" else [args.profile]
    outputs = {profile: ROOT / args.output_dir / profile for profile in profiles}
    for output in outputs.values():
        if output.exists() or output.is_symlink():
            parser.error(f"输出目录已存在，请换一个 --output-dir：{output}")

    for profile, output in outputs.items():
        lr_scale = args.lr_scale if args.lr_scale is not None else LR_SCALE_DEFAULTS[profile]
        report = train_profile(
            profile, output, epochs=args.epochs, lr_scale=lr_scale, batch_size=args.batch_size,
            patience=args.patience, resume=ROOT / args.resume if args.resume is not None else None,
            keep_best_continued=args.keep_best_continued, weight_decay=args.weight_decay,
        )
        metric = report["primary_metric"]
        before = report["validation_before"][metric]
        after = report["validation_after"][metric]
        if report["retained_parent"]:
            result = "未超过父检查点，本轮保留父检查点"
        else:
            result = "主指标改善" if report["improved_primary_metric"] else "主指标未改善（按 --keep-best-continued 保存续训候选）"
        print(f"{profile}: {metric} {before:.6f} -> {after:.6f}（{result}）", flush=True)
        print(f"  候选和日志：{output}", flush=True)


if __name__ == "__main__":
    main()
