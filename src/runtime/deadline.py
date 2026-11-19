"""硬超时（T13）：单调时钟，不随系统时间跳变。"""
from __future__ import annotations

import time


class Deadline:
    def __init__(self, budget_seconds: float):
        if budget_seconds <= 0:
            raise ValueError("budget_seconds 必须大于 0")
        self.budget = float(budget_seconds)
        self._end = time.monotonic() + self.budget

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def check(self, stage: str = "") -> None:
        if self.expired():
            raise TimeoutError(f"超过时间预算 {self.budget:.0f}s" + (f"（{stage}）" if stage else ""))
