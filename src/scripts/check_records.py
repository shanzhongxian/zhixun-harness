"""记录完整性自检（T27）：缺字段即失败。"""
import argparse
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent   # src/
ROOT = SRC.parent                              # 包根目录（智训引擎/）
sys.path.insert(0, str(SRC))

RUNS = Path(os.environ.get("HARNESS_RUNS_DIR", str(ROOT / "runs")))

REQUIRED_TOP = ["profile", "strategy", "round", "action", "hypothesis", "evidence_ids",
                "params", "decision_source", "result", "current_checkpoint", "new_checkpoint", "diagnostics"]
REQUIRED_DIAG = ["primary_metric", "direction", "validation_metrics", "train_metrics",
                 "overfit_flag", "drift_flag", "gap", "n_groups"]


def check(path):
    path = Path(path)
    if not path.exists():
        return False, [f"缺少文件 {path}"]
    errors = []
    ok = True
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                ok = False
                errors.append(f"第 {i} 行 JSON 非法: {exc}")
                continue
            for key in REQUIRED_TOP:
                if key not in rec:
                    ok = False
                    errors.append(f"第 {i} 行缺字段 {key}")
            diag = rec.get("diagnostics", {})
            for key in REQUIRED_DIAG:
                if key not in diag:
                    ok = False
                    errors.append(f"第 {i} 行 diagnostics 缺 {key}")
            if rec.get("action") == "train":
                for key in ("replay_split", "replay_metric", "candidate_report"):
                    if key not in rec:
                        ok = False
                        errors.append(f"第 {i} 行 train 记录缺 {key}")
    return ok, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(RUNS / "decisions.jsonl"))
    args = parser.parse_args()
    ok, errors = check(args.records)
    if ok:
        print(f"PASS 记录完整：{args.records}")
        return 0
    print(f"FAIL 记录不完整：{args.records}")
    for e in errors:
        print("  -", e)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
