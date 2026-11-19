"""策略版本库：policies/pNNN.json + champion 指针，支持提交与回滚。"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .spec import PolicySpec


class PolicyStore:
    def __init__(self, root):
        self.root = Path(root)
        self.dir = self.root / "policies"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pointer = self.dir / "champion.json"

    def _ids(self) -> list:
        return sorted(p.stem for p in self.dir.glob("p*.json"))

    def next_id(self) -> str:
        ids = [int(re.sub(r"\D", "", i)) for i in self._ids() if re.sub(r"\D", "", i)]
        return f"p{(max(ids) + 1) if ids else 1:03d}"

    def save(self, policy: PolicySpec) -> Path:
        return policy.save(self.dir / f"{policy.id}.json")

    def load(self, policy_id: str) -> PolicySpec:
        return PolicySpec.load(self.dir / f"{policy_id}.json")

    def champion(self) -> PolicySpec | None:
        if not self.pointer.exists():
            return None
        return self.load(json.loads(self.pointer.read_text(encoding="utf-8"))["id"])

    def commit(self, policy: PolicySpec) -> None:
        policy.status = "champion"
        for other in self._ids():
            spec = self.load(other)
            if spec.status == "champion":
                spec.status = "retired"
                self.save(spec)
        self.save(policy)
        self.pointer.write_text(json.dumps({"id": policy.id}, ensure_ascii=False) + "\n", encoding="utf-8")

    def rollback(self, policy_id: str) -> None:
        self.commit(self.load(policy_id))

    def history(self) -> list:
        return [self.load(i) for i in self._ids()]
