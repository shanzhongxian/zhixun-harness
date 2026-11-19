"""产物写出：submit/round-N/<profile>/，原子写 + sha256 + no_op。

命名与仓库既有风格一致：round-N 目录、checkpoint.npz、artifact.json、no_op.json。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_write_bytes(path, payload: bytes) -> Path:
    """先写临时文件并 fsync，再 os.replace，保证平台永远读不到半成品。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def atomic_write_json(path, obj) -> Path:
    return atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _target_dir(submit_dir, profile: str, round_no: int) -> Path:
    return Path(submit_dir) / f"round-{round_no}" / profile


def write_artifact(submit_dir, profile: str, round_no: int, checkpoint, *,
                   primary_metric: str, value: float, gate: str,
                   training_json=None, extra: dict | None = None) -> Path:
    """上线产物：复制 checkpoint + 写 artifact.json（含 sha256 与门禁结论）。"""
    target = _target_dir(submit_dir, profile, round_no)
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "checkpoint.npz"
    atomic_write_bytes(destination, Path(checkpoint).read_bytes())
    payload = {
        "profile": profile,
        "round": round_no,
        "checkpoint": project_relative(destination),
        "sha256": sha256_file(destination),
        "primary_metric": primary_metric,
        "value": value,
        "gate": gate,
        "training_json": project_relative(training_json) if training_json else None,
    }
    if extra:
        payload.update(extra)
    atomic_write_json(target / "artifact.json", payload)
    return target


def write_no_op(submit_dir, profile: str, round_no: int, *,
                current_checkpoint=None, hypothesis: str = "", reason: str = "no_op",
                extra: dict | None = None) -> Path:
    """不上线：只写 no_op.json，不动当前版本指针。"""
    target = _target_dir(submit_dir, profile, round_no)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile,
        "round": round_no,
        "deploy": False,
        "reason": reason,
        "hypothesis": hypothesis,
        "current_checkpoint": project_relative(current_checkpoint) if current_checkpoint else None,
    }
    if extra:
        payload.update(extra)
    atomic_write_json(target / "no_op.json", payload)
    return target


def project_relative(path) -> str:
    """产物里只写相对仓库根的路径，换机器不失效。"""
    import project

    return project.to_store_path(path)
