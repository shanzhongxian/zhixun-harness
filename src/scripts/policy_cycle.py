"""策略进化一轮（Policy Arena 的最小闭环）。

流程：
  读历史决策 → 冠军打分（回放或真实训练探针）→ 提案器生成候选 →
  竞技场打分 →（阈值放宽则过元门禁）→ 提交或拒绝 → 打印报告

不训练主模型（探针训练到临时目录，用完即删）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from policy.arena import load_records, score  # noqa: E402
from policy.merge import merge  # noqa: E402
from policy.meta_gate import MetaGate  # noqa: E402
from policy.proposers import BanditProposer, CalibrateProposer  # noqa: E402
from policy.spec import PolicySpec  # noqa: E402


def bootstrap(store) -> PolicySpec:
    spec = PolicySpec(id=store.next_id(), created_by="bootstrap")
    store.commit(spec)
    return spec


def _kwargs(spec: PolicySpec, profile: str) -> dict:
    s = spec.search_for(profile)
    return {
        "lr_scale": s["lr_scale"],
        "epochs": s["epochs"],
        "patience": s["patience"],
        "weight_decay": s.get("weight_decay", 0.0),
    }


def anchored_false_accept(policy: PolicySpec, records: list) -> float:
    """锚定集：固定取前 30% 记录，衡量“该拒却收”的比例。"""
    if not records:
        return 0.0
    cut = max(1, int(len(records) * 0.3))
    stat = score(policy, records[:cut])
    judged = stat["accepted"] + stat["rejected"]
    return stat["accepted"] / judged if judged else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--propose", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="只演示，不写入策略库")
    parser.add_argument("--merge", action="store_true", help="额外尝试合并两个候选")
    parser.add_argument("--train-profile", choices=("", "A", "B", "C"), default="",
                        help="用真实训练探针评估参数类候选（建议 A，几秒钟）")
    parser.add_argument("--train-base", choices=("current", "baseline"), default="current",
                        help="探针起点：current=当前冠军续训；baseline=从随包基线重新探索")
    args = parser.parse_args()

    root = Path(args.root)
    from policy.store import PolicyStore

    store = PolicyStore(root)
    records = load_records(root / args.runs)
    if not records and not args.train_profile:
        print("没有可用的历史决策点（需要 diagnostics + candidate_metric）")
        return 1

    champion = store.champion() or bootstrap(store)
    train_profile = args.train_profile

    if train_profile:
        from policy import train_eval
        base = train_eval.probe(root, train_profile, base=args.train_base, **_kwargs(champion, train_profile))
        print(f"冠军 {champion.id}（真实训练探针 {train_profile}/{args.train_base}）：{base}")
    else:
        base = score(champion, records)
        print(f"冠军 {champion.id}：{base}")

    proposals = BanditProposer(seed=args.round).propose(champion, k=args.propose)
    if not train_profile:
        proposals += CalibrateProposer().propose(champion, records)

    base_id = int(store.next_id().lstrip("p"))
    results = []
    for offset, spec in enumerate(proposals):
        spec.id = f"p{base_id + offset:03d}"
        if train_profile:
            stat = train_eval.probe(root, train_profile, base=args.train_base, **_kwargs(spec, train_profile))
        else:
            stat = score(spec, records)
        results.append((spec, stat))
        print(f"候选 {spec.id:<5} by {spec.created_by:<9} {stat}")

    results.sort(key=lambda x: x[1]["score"], reverse=True)
    gate = MetaGate(root)
    chosen = None
    for spec, stat in results:
        if stat["score"] <= base["score"] + 1e-12:
            continue
        loose = spec.loose_fields(champion)
        if loose:
            ok, reasons = gate.check_thresholds(
                champion.thresholds, spec.thresholds, round_no=args.round,
                anchored_false_accept=anchored_false_accept(spec, records),
                anchored_baseline=anchored_false_accept(champion, records),
            )
            if not ok:
                print(f"候选 {spec.id} 被元门禁拒绝（{loose}）：{reasons}")
                continue
            print(f"候选 {spec.id} 通过元门禁（放宽字段：{loose}）")
        chosen = (spec, stat)
        break

    if chosen is None:
        print(f"\n拒绝：没有候选同时满足「收益提升」与「元门禁」，冠军保持 {champion.id}")
        if train_profile:
            print("提示：真实训练探针显示当前参数已到平台期，建议换起点（--train-base baseline）或换候选网格。")
        else:
            print("提示：参数类候选请加 --train-profile A 走真实训练探针。")
        return 0

    best, best_stat = chosen

    if args.merge and not train_profile and len(results) >= 2:
        second = results[1][0]
        merged, info = merge(champion, second, records)
        if merged:
            merged.id = store.next_id()
            merged_stat = score(merged, records)
            print(f"合并候选 {merged.id}：{merged_stat}")
            if merged_stat["score"] > best_stat["score"]:
                best, best_stat = merged, merged_stat

    print(f"\n提交冠军：{best.id}（{base['score']} → {best_stat['score']}）")
    if args.dry_run:
        print("dry-run：未写入策略库")
        return 0
    store.commit(best)
    from policy import trace
    event = trace.log_event(root, best, best.diff(champion), note=f"round {args.round}")
    print(f"已提交，策略文件：policies/{best.id}.json")
    print(f"演化记录：policies/events.jsonl（SwanLab: {event.get('swanlab')}）")
    print(json.dumps(best.to_dict(), ensure_ascii=False, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
