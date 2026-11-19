"""数据适配：行的展开规则直接复用基线包，保证训练/推理特征顺序完全一致。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:  # 镜像里 baselines 与 app 同目录
    from baselines.data import flatten, json_value
except ImportError:  # 本地联调
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "cpu-a-guide-20261110-v1"))
    from baselines.data import flatten, json_value

__all__ = ["flatten", "json_value", "vector", "write_jsonl", "dataset_key", "fingerprint"]


def vector(inputs: Dict[str, Any], paths: Sequence[str]) -> List[float]:
    """推理期按训练期的路径表取值，缺失补 0，避免顺序漂移。"""
    table = dict(zip(*flatten(inputs)))
    return [table.get(path, 0.0) for path in paths]


def write_jsonl(rows: Iterable[Dict[str, Any]], target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, dict) or "inputs" not in row or "answer" not in row:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def dataset_key(raw: Any, default: str = "A") -> str:
    text = str(raw or "").strip().upper()
    if text in ("A", "B", "C"):
        return text
    for token in re.split(r"[^A-Z0-9]+", text):
        if token in ("A", "B", "C"):
            return token
    return default


def fingerprint(paths: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:16]
