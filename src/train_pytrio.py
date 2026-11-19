"""PyTRiO 接入（原 训练.py）：默认只探测，不训练、不生成。

用法（仓库根目录）：
    python src/train_pytrio.py --probe                 # 检查凭据 + qwen3.8-27b 是否可建客户端
    python src/train_pytrio.py --probe --generate      # 额外做一次 ≤32 token 的推理探针（耗 token）
    python src/train_pytrio.py --run-training          # 预留开关：首次训练，当前被显式拒绝

设计原则：任何会改动模型或产生训练产物的动作，都必须显式加开关，
默认行为永远是「只读探测」，避免误触发正式训练。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from pytrio_client import DEFAULT_MODEL, generate, probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="只探测凭据与模型可用性")
    parser.add_argument("--generate", action="store_true", help="探测通过后做一次极短推理")
    parser.add_argument("--model", default=os.environ.get("PYTRIO_MODEL", DEFAULT_MODEL),
                        help=f"模型 ID，默认 {DEFAULT_MODEL}")
    parser.add_argument("--prompt", default="只回复两个字：就绪", help="探测用的提示词")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--run-training", action="store_true", help="预留：开始首次训练（当前拒绝）")
    args = parser.parse_args()

    if args.run_training:
        print("拒绝执行：首次训练尚未批准。")
        print("批准后需要：1) 接好批次数据 -> harness.run_round；2) 加 --yes 二次确认；3) 记录到 SwanLab。")
        return 3

    if not args.probe and not args.generate:
        args.probe = True

    result = probe(args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        print("\n探测未通过，未执行任何推理或训练。")
        return 2

    if args.generate:
        generated = generate(args.prompt, model_id=args.model, max_tokens=args.max_tokens)
        print(json.dumps(generated, ensure_ascii=False, indent=2))

    print("\n探测通过：模型可用。训练动作仍处于暂停状态（--run-training 未批准）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
