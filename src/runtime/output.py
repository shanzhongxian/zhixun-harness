"""兜底输出（T13）：任何异常都要留下一份合法 result.json。

平台只认产物，不认日志；这里保证「崩了也有话可说」。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def fallback_result(reason: str, **extra) -> dict:
    payload = {"ok": False, "reason": reason, "stage": "fallback"}
    payload.update(extra)
    return payload


def write_fallback(out_dir, reason: str, **extra) -> Path:
    out_dir = Path(out_dir)
    return atomic_write_json(out_dir / "result.json", fallback_result(reason, **extra))
