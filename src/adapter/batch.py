"""批次读取：inbox/batch-XXXX.jsonl → 与 read_dataset 完全一致的结构。

批次文件就是公开数据同款 JSONL（id / inputs / answer / meta），
因此直接复用 project.read_dataset，不另写一套解析，避免两套 schema 漂移。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import project

DEFAULT_GLOB = "batch-*.jsonl"
MAX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class BatchRef:
    batch_id: str
    path: Path
    size_bytes: int
    manifest: dict = field(default_factory=dict)


def _manifest_for(path: Path) -> dict:
    """可选旁路元数据：batch-0001.meta.json；没有就按文件名推断。"""
    sidecar = path.with_suffix(".meta.json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def iter_batches(batch_dir, patterns=(DEFAULT_GLOB,)) -> list[BatchRef]:
    """按文件名排序返回批次；跳号是允许的（B 包按批次下发，可能不连续）。"""
    batch_dir = Path(batch_dir)
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"批次目录不存在：{batch_dir}")
    refs: list[BatchRef] = []
    for pattern in patterns:
        for path in sorted(batch_dir.glob(pattern)):
            if not path.is_file() or path.is_symlink():
                continue
            refs.append(BatchRef(path.stem, path, path.stat().st_size, _manifest_for(path)))
    if not refs:
        raise FileNotFoundError(f"{batch_dir} 下没有匹配 {patterns} 的批次文件")
    return refs


def read_batch(ref: BatchRef, task: str, classes=None, expected_paths=None):
    """校验并读出一个批次，返回 (数据字典, 批次元信息)。"""
    if ref.size_bytes > MAX_BYTES:
        raise ValueError(f"{ref.path.name} 超过 128MiB 上限")
    data = project.read_dataset(ref.path, task, classes, expected_paths)
    info = {
        "batch_id": ref.batch_id,
        "path": project.to_store_path(ref.path),
        "n_rows": len(data["ids"]),
        "sha256": data["sha256"],
        "manifest": ref.manifest,
    }
    return data, info
