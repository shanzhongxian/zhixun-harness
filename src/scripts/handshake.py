"""握手 demo（T02）：--mock 用公开验证集切片造 3 批；--real 读真实批次目录。

默认只做读取 + 校验 + 写 no_op 产物，不训练。

用法（在仓库根目录执行）：
    python src/scripts/handshake.py --mock --profile A --batches 3
    python src/scripts/handshake.py --real --profile A --batch-dir inbox --submit-dir submit
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent   # src/
ROOT = SRC.parent                              # 仓库根（智训引擎/）
sys.path.insert(0, str(SRC))

import project  # noqa: E402
from adapter.platform import run_batches  # noqa: E402


def make_mock_batches(profile: str, batches: int, out_dir: Path) -> Path:
    """把公开训练集切成 N 段，模拟平台按批下发（字段与真实批次完全一致）。

    用训练集切片而不是验证集，是为了训练分支（--train）时不会和验证集 ID 重叠。
    """
    source = project.PACKAGES[profile] / "public" / "train.jsonl"
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < batches:
        raise SystemExit(f"样本太少（{len(lines)} 行），无法切成 {batches} 批")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("batch-*.jsonl"):
        stale.unlink()
    size = -(-len(lines) // batches)  # ceil
    for index in range(batches):
        chunk = lines[index * size:(index + 1) * size]
        if not chunk:
            continue
        (out_dir / f"batch-{index + 1:04d}.jsonl").write_text("\n".join(chunk) + "\n", encoding="utf-8")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("A", "B", "C"), default="A")
    parser.add_argument("--mock", action="store_true", help="用公开验证集切片造批次")
    parser.add_argument("--real", action="store_true", help="读真实批次目录")
    parser.add_argument("--batches", type=int, default=3, help="mock 批次数量")
    parser.add_argument("--batch-dir", default=None)
    parser.add_argument("--submit-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少批")
    parser.add_argument("--train", action="store_true", help="启用训练分支（批次→续训→门禁→产物）")
    parser.add_argument("--strategy", choices=("rule", "mock", "qwen"), default="rule")
    parser.add_argument("--clean", action="store_true", help="mock 前清空 .selftest/inbox 与 submit")
    args = parser.parse_args()

    if not args.mock and not args.real:
        parser.error("必须指定 --mock 或 --real")

    if args.mock:
        batch_dir = Path(args.batch_dir or (ROOT / ".selftest" / "inbox"))
        submit_dir = Path(args.submit_dir or (ROOT / ".selftest" / "submit"))
        if args.clean and submit_dir.exists():
            shutil.rmtree(submit_dir)
        make_mock_batches(args.profile, args.batches, batch_dir)
    else:
        if not args.batch_dir:
            parser.error("--real 需要 --batch-dir")
        batch_dir = Path(args.batch_dir)
        submit_dir = Path(args.submit_dir or (ROOT / "submit"))

    results = run_batches(batch_dir, submit_dir, args.profile, strategy=args.strategy,
                          allow_train=args.train, limit=args.limit)

    print(f"profile={args.profile} batch_dir={batch_dir}  mode={'train' if args.train else 'handshake'}")
    print(f"{'batch':<12}{'rows':>6}  {'action':<8}{'result':<10}submit")
    for row in results:
        print(f"{row['batch_id']:<12}{row['rows']:>6}  {row.get('action', '-'):<8}{row.get('result', '-'):<10}{row['submit']}")
    print(f"\n完成：{len(results)} 批（{'训练+门禁' if args.train else '仅握手 no_op'}）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
