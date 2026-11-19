"""统一路径出口：镜像里只有 /workspace 可写，其余只读。"""
from __future__ import annotations

import os
from pathlib import Path

APP = Path("/app")
DATA = Path("/data")
PREVIOUS = Path("/previous")
STATE = Path("/state")
MODEL = Path("/model")
WORKSPACE = Path(os.environ.get("CONTEST_WORK", "/workspace"))

# 本地联调时允许覆盖（评测环境不会设置）
LOCAL_ROOT = os.environ.get("CONTEST_LOCAL_ROOT")


def _root(path: Path) -> Path:
    if LOCAL_ROOT:
        return Path(LOCAL_ROOT) / path.as_posix().lstrip("/")
    return path


def data_dir() -> Path:
    return _root(DATA)


def previous_dir() -> Path:
    return _root(PREVIOUS)


def state_dir() -> Path:
    return _root(STATE)


def model_dir() -> Path:
    return _root(MODEL)


def workspace() -> Path:
    """可写工作区：优先 CONTEST_WORK（默认 /workspace），建不出来或写不进去就退回 /tmp。"""
    path = _root(WORKSPACE)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return path
    except OSError:
        fallback = _root(Path("/tmp/workspace"))
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def round_dir(round_index: int) -> Path:
    path = workspace() / "rounds" / f"round-{round_index}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
