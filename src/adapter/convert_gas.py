"""A 包原始气体批次 → 批次 JSONL（T03）。

原始格式：`batchN.dat` 每行 129 个字段 = 类别标签(1–6) + 128 个 `编号:数值`。
输出格式：与 `public/*.jsonl` 完全一致（id / inputs.features / answer / meta）。

用法（仓库根目录）：
    python src/adapter/convert_gas.py --source "<gas 数据集 Dataset 目录>" --out inbox_gas
    python src/adapter/convert_gas.py --source ... --out inbox_gas --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
FEATURES = 128


def parse_line(line: str, batch_no: int, index: int) -> dict:
    parts = line.split()
    if len(parts) != FEATURES + 1:
        raise ValueError(f"字段数应为 {FEATURES + 1}，实际 {len(parts)}")
    label = int(parts[0])
    if not 1 <= label <= 6:
        raise ValueError(f"类别标签越界：{label}")
    pairs = {}
    for token in parts[1:]:
        key, value = token.split(":")
        pairs[int(key)] = float(value)
    if sorted(pairs) != list(range(1, FEATURES + 1)):
        raise ValueError("特征编号缺失或重复")
    return {
        "id": f"gas_b{batch_no:02d}_{index:06d}",
        "inputs": {"features": [pairs[i] for i in range(1, FEATURES + 1)]},
        "answer": CLASSES[label - 1],
        "meta": {"source_batch": batch_no, "source_index": index},
    }


def convert(source_dir: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for path in sorted(source_dir.glob("batch*.dat"), key=lambda p: int(p.stem.replace("batch", ""))):
        batch_no = int(path.stem.replace("batch", ""))
        records = []
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for index, line in enumerate(fh, 1):
                if line.strip():
                    records.append(parse_line(line, batch_no, index))
        target = out_dir / f"batch-{batch_no:04d}.jsonl"
        with target.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats.append({"batch": batch_no, "rows": len(records), "file": target.name})
    return stats


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def verify(out_dir: Path, package_dir: Path) -> bool:
    """batch-0001/0002 必须与官方 public 的 train/validation 完全一致。"""
    pairs = [(1, "train"), (2, "validation")]
    ok = True
    for batch_no, split in pairs:
        mine = load_jsonl(out_dir / f"batch-{batch_no:04d}.jsonl")
        official = load_jsonl(package_dir / "public" / f"{split}.jsonl")
        same_len = len(mine) == len(official)
        same_ids = [r["id"] for r in mine] == [r["id"] for r in official]
        same_ans = [r["answer"] for r in mine] == [r["answer"] for r in official]
        same_feat = all(a["inputs"]["features"] == b["inputs"]["features"]
                        for a, b in zip(mine, official))
        flag = same_len and same_ids and same_ans and same_feat
        ok &= flag
        print(f"  batch-{batch_no:04d} vs {split}: rows={len(mine)}/{len(official)} "
              f"ids={'OK' if same_ids else 'DIFF'} answer={'OK' if same_ans else 'DIFF'} "
              f"features={'OK' if same_feat else 'DIFF'} -> {'PASS' if flag else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="含 batch1.dat..batch10.dat 的目录")
    parser.add_argument("--out", default="inbox_gas", help="输出目录")
    parser.add_argument("--package", default=None, help="官方训练包目录（--verify 用）")
    parser.add_argument("--verify", action="store_true", help="与 public 逐位比对 batch1/2")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    stats = convert(source, out)
    total = sum(s["rows"] for s in stats)
    print(f"写出 {len(stats)} 个批次，共 {total} 行 → {out}")
    for s in stats:
        print(f"  batch-{s['batch']:04d}: {s['rows']} 行")
    if args.verify:
        package = Path(args.package or (Path(__file__).resolve().parent.parent.parent
                                        / "packages" / "cpu-a-guide-20261110-v1"))
        print("逐位比对：")
        return 0 if verify(out, package) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
