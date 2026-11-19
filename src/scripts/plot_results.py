"""绘图脚本：从 runs/ 的决策日志与训练历史生成图表，按工况分类保存到 plots/。

用法：
    python scripts/plot_results.py                  # 读默认 runs/，输出 plots/
    python scripts/plot_results.py --records .x/runs/decisions.jsonl --out .x/plots

输出（新文件夹 plots/，按工况 A/B/C 分类）：
    plots/<profile>/<profile>_training_curve.png   训练 loss + 验证主指标曲线
    plots/<profile>/<profile>_group_metrics.png    分组（批次/轴承/发动机）主指标柱状图
    plots/summary.md                               汇总

依赖 matplotlib（系统 Python 已装）。断网/无 matplotlib 时本脚本不可用，不影响 harness 本体。
"""
import argparse
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent   # src/
ROOT = SRC.parent                              # 包根目录（智训引擎/）
sys.path.insert(0, str(SRC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategy.schemas import PRIMARY_METRIC, DIRECTION

RUNS = Path(os.environ.get("HARNESS_RUNS_DIR", str(ROOT / "runs")))


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_history(profile):
    """在 RUNS 与 ROOT/runs 下递归找该 profile 的 history.jsonl。"""
    found = []
    for base in (RUNS, ROOT / "runs"):
        if base.exists():
            for p in base.rglob("history.jsonl"):
                if p.parent.name == profile:
                    found.append(p)
    return sorted(set(found))


def plot_training_curve(profile, out_dir):
    records = []
    for path in find_history(profile):
        records.extend(load_jsonl(path))
    if not records:
        print(f"  [{profile}] 无 history.jsonl，跳过训练曲线")
        return
    records.sort(key=lambda r: (r.get("epoch", 0), r.get("step", 0)))
    epochs = [r["epoch"] for r in records]
    losses = [r["loss"] for r in records]
    primary = PRIMARY_METRIC[profile]
    vals = [r["validation"][primary] for r in records]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, losses, marker="o", color="tab:red", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(epochs, vals, marker="s", color="tab:blue", label=primary)
    ax2.set_ylabel(primary, color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    direction = "higher is better" if DIRECTION[profile] > 0 else "lower is better"
    fig.suptitle(f"[{profile}] training curve  ({primary}, {direction})")
    fig.tight_layout()
    out = out_dir / f"{profile}_training_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  已保存 {out}")


def plot_group_metrics(profile, records, out_dir):
    latest = None
    for r in reversed(records):
        if r["profile"] == profile and r.get("diagnostics", {}).get("group_metrics"):
            latest = r
            break
    if not latest:
        print(f"  [{profile}] 无分组指标，跳过")
        return
    groups = latest["diagnostics"]["group_metrics"]
    labels = [str(tuple(g["group"])) for g in groups]
    values = [g["primary"] for g in groups]
    sizes = [g["n"] for g in groups]

    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(groups)), 5))
    bars = ax.bar(range(len(groups)), values)
    for i, (v, n) in enumerate(zip(values, sizes)):
        ax.text(i, v, f"{v:.1f}\n(n={n})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(PRIMARY_METRIC[profile])
    ax.set_title(f"[{profile}] group metric - current model ({len(groups)} groups)")
    fig.tight_layout()
    out = out_dir / f"{profile}_group_metrics.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  已保存 {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(RUNS / "decisions.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "plots"))
    args = parser.parse_args()

    records = load_jsonl(args.records)
    profiles = sorted({r["profile"] for r in records}) or ["A", "B", "C"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        pdir = out_dir / profile
        pdir.mkdir(parents=True, exist_ok=True)
        plot_training_curve(profile, pdir)
        plot_group_metrics(profile, records, pdir)

    lines = ["# 绘图汇总", ""]
    for profile in profiles:
        latest = next((r for r in reversed(records) if r["profile"] == profile), None)
        if latest:
            p = PRIMARY_METRIC[profile]
            before = latest["diagnostics"]["validation_metrics"][p]
            if "candidate_metric" in latest and p in latest["candidate_metric"]:
                after = latest["candidate_metric"][p]
                lines.append(f"- **{profile}**: {p} {before:.6f} -> {after:.6f}, "
                             f"result={latest['result']}, round={latest.get('round')}")
            else:
                lines.append(f"- **{profile}**: {p}={before:.6f}, "
                             f"result={latest['result']}, round={latest.get('round')}")
    lines.append("")
    lines.append("图表按工况（A/B/C）分类保存在本目录下各子目录。")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总已保存 {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
