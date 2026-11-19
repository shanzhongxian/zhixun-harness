"""离线记录 → 赛后上传（T26）：本地 swanlog 一键同步到 SwanLab。

评测环境无外网时用 SWANLAB_MODE=local 只写本地 swanlog；赛后联网执行本脚本
把本地实验同步到远端（swanlab.cn/@shanzhongxian/zhixun-engine）。

用法（仓库根目录）：
    python src/scripts/sync_swanlab.py
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    try:
        subprocess.run([sys.executable, "-m", "swanlab", "sync"], check=True)
        print("同步完成。")
        return 0
    except FileNotFoundError:
        print("未找到 swanlab，请先：python -m pip install swanlab")
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"同步失败（退出码 {exc.returncode}）。")
        print("请确认：1) 已联网；2) 已登录（swanlab login 或设置 SWANLAB_API_KEY）；")
        print("       3) 目标项目 zhixun-engine@shanzhongxian 存在。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
