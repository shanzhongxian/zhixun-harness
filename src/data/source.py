"""双包数据隔离（#2）。

规则：
- A（公开包）与 B（隐藏包）各有独立根目录、独立 inbox/submit；
- 读取走两套入口（Source.read_list），写入也只写自己的 submit；
- 但策略是同一套：最终由 `unified_plan()` 归元成一份迭代方案。

评测环境里 B 包由平台按批下发，这里只预留目录与接口，不会伪造数据。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    root: Path

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def submit(self) -> Path:
        return self.root / "submit"

    def prepare(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.submit.mkdir(parents=True, exist_ok=True)
        marker = self.root / ".source"
        if not marker.exists():
            marker.write_text(self.name + "\n", encoding="utf-8")

    def read_list(self) -> list:
        """本包可读的批次清单；B 包为空是正常的（平台下发后才有）。"""
        if not self.inbox.is_dir():
            return []
        return sorted(p.name for p in self.inbox.glob("batch-*.jsonl"))

    def resolve(self, relative: str) -> Path:
        """防目录穿越：只允许落在本源目录内。"""
        target = (self.root / relative).resolve()
        root = self.root.resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"{self.name} 源越界：{relative}")
        return target


class DualSource:
    def __init__(self, base):
        base = Path(base)
        self.A = Source("A", base / "data" / "A")
        self.B = Source("B", base / "data" / "B")

    def prepare(self) -> None:
        for src in (self.A, self.B):
            src.prepare()

    def assert_isolated(self) -> None:
        a, b = self.A.root.resolve(), self.B.root.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("A/B 数据根目录发生重叠，隔离失效")

    def unified_plan(self, policy_id: str) -> dict:
        """两个源、两套读写，但只有一份迭代方案。"""
        self.assert_isolated()
        return {
            "policy_id": policy_id,
            "sources": [
                {"name": "A", "root": str(self.A.root), "batches": len(self.A.read_list()), "submit": str(self.A.submit)},
                {"name": "B", "root": str(self.B.root), "batches": len(self.B.read_list()), "submit": str(self.B.submit)},
            ],
            "note": "策略共用、数据隔离；B 包在评测时按批下发",
        }
