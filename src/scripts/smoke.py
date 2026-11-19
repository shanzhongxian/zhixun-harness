"""冒烟测试（T14）：只验证链路，不训练。

用法（仓库根目录）：python src/scripts/smoke.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))


def run(name: str, args: list[str]) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)
    ok = proc.returncode == 0
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return ok, (tail[-1] if tail else "")


def main() -> int:
    checks = []

    ok, msg = run("adapter 单测", [str(SRC / "tests" / "test_adapter.py")])
    checks.append(("adapter 单测", ok, msg.splitlines()[-1] if msg else ""))

    ok, msg = run("握手 mock 3 批", [str(SRC / "scripts" / "handshake.py"),
                                     "--mock", "--profile", "A", "--batches", "3",
                                     "--submit-dir", str(Path(tempfile.gettempdir()) / "smoke_submit")])
    checks.append(("握手 mock", ok, "3 批通过" if ok else msg))

    gas_dir = ROOT / "inbox_gas"
    if gas_dir.is_dir():
        ok, msg = run("握手 real 10 批", [str(SRC / "scripts" / "handshake.py"),
                                          "--real", "--profile", "A",
                                          "--batch-dir", str(gas_dir),
                                          "--submit-dir", str(Path(tempfile.gettempdir()) / "smoke_submit_real")])
        checks.append(("握手 real", ok, "10 批通过" if ok else msg))
    else:
        checks.append(("握手 real", None, "未生成 inbox_gas，跳过"))

    decisions = ROOT / "runs" / "decisions.jsonl"
    if decisions.exists():
        ok, msg = run("记录完整性", [str(SRC / "scripts" / "check_records.py")])
        checks.append(("记录完整性", ok, msg))
    else:
        checks.append(("记录完整性", None, "无 runs/decisions.jsonl，跳过"))

    print(f"{'检查项':<16}{'结果':<8}说明")
    failed = 0
    for name, ok, note in checks:
        label = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        failed += 0 if ok is not False else 1
        print(f"{name:<16}{label:<8}{note}")
    print("\n冒烟完成：全部通过（未触发训练）" if failed == 0 else f"\n{failed} 项失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
