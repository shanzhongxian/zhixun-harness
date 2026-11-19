"""回滚演练（T24）：验证回滚到 previous 版本 <30s 且模型可加载、指标可重算。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent   # src/
ROOT = SRC.parent                              # 包根目录（智训引擎/）
sys.path.insert(0, str(SRC))
import project
from strategy.schemas import PRIMARY_METRIC

RUNS = Path(os.environ.get("HARNESS_RUNS_DIR", str(ROOT / "runs")))


def _resolve(value):
    """相对路径按仓库根解析，兼容旧版本写死的绝对路径。"""
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    parts = path.parts
    for anchor in ("runs", "packages", ".selftest"):
        if anchor in parts:
            candidate = ROOT.joinpath(*parts[parts.index(anchor):])
            if candidate.exists():
                return candidate
    return path


def primary_of(profile, checkpoint):
    config = json.loads((project.PACKAGES[profile] / "baselines" / "configs" / f"{profile}.json").read_text(encoding="utf-8"))
    model = project.Model.load(checkpoint)
    validation = project.read_dataset(
        project.PACKAGES[profile] / "public" / "validation.jsonl",
        config["task"], config.get("classes", []), model.meta["paths"],
    )
    return project.evaluate(profile, model, validation, nasa=(profile == "C"))[PRIMARY_METRIC[profile]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()

    current_json = RUNS / "current.json"
    if not current_json.exists():
        print(f"FAIL 缺少 {current_json}（先跑一次 harness 生成 current.json）")
        return 1
    current = json.loads(current_json.read_text(encoding="utf-8"))
    all_ok = True
    for profile, entry in current.items():
        prev = _resolve(entry.get("previous"))
        if not prev:
            print(f"[{profile}] 无 previous（尚未产生回滚目标），跳过")
            continue
        t0 = time.monotonic()
        metric = primary_of(profile, prev)
        dt = time.monotonic() - t0
        ok = dt <= args.max_seconds
        all_ok &= ok
        print(f"[{profile}] 回滚到 previous：重载+重评 {dt:.3f}s，{PRIMARY_METRIC[profile]}={metric:.6f}，"
              f"{'PASS' if ok else f'FAIL(>{args.max_seconds}s)'}")
    print("回滚演练 PASS" if all_ok else "回滚演练 FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
