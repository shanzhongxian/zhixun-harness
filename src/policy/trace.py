"""策略演化记录（#3）：每次都写本地 events.jsonl，联网时再上报 SwanLab。

原则：
- 本地一定写（评测环境无网也能留证）；
- SwanLab 用独立 run（experiment_name = policy-<id>），不挤在训练 run 里；
- 只记摘要（id/parent/阈值/参数），不记 skills/code 全文。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


def _local_event(root: Path, payload: dict) -> Path:
    path = Path(root) / "policies" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def log_event(root, policy, diff: dict | None = None, note: str = "") -> dict:
    root = Path(root)
    version = int(re.sub(r"\D", "", policy.id) or 0)
    payload = {
        "ts": round(time.time(), 3),
        "policy_id": policy.id,
        "version": version,
        "parent": policy.parent,
        "created_by": policy.created_by,
        "note": note,
        "summary": policy.summary(),
        "diff_keys": sorted((diff or {}).keys()),
    }
    _local_event(root, payload)

    try:
        import swanlab

        cfg = json.loads((root / "configs" / "platform.json").read_text(encoding="utf-8")).get("swanlab", {})
        run = swanlab.init(
            project=cfg.get("project", "zhixun-engine"),
            workspace=cfg.get("workspace"),
            experiment_name=f"policy-{policy.id}",
            config=payload["summary"],
            public=bool(cfg.get("public", True)),
        )
        swanlab.log({"policy/version": version}, step=0)
        swanlab.finish()
        payload["swanlab"] = "uploaded"
    except Exception as error:  # noqa: BLE001 - 上报失败不能影响主流程
        payload["swanlab"] = f"skipped: {type(error).__name__}"
        _local_event(root, {**payload, "kind": "swanlab_error"})
    return payload
