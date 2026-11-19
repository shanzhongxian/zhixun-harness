"""SwanLab 打点（T25/T26）：把工业模型训练过程记录到 SwanLab，评委可查。

原则：
- SwanLab 未安装 / 初始化失败 / 断网时静默跳过，绝不阻塞训练；
- 离线（SWANLAB_MODE=local）时只写本地 swanlog，赛后 `swanlab sync` 上传；
- 每个 round 一个实验，experiment_name = f"{profile}-round{n}"，按数据类型分到 SwanLab 的组：
  A=Gas（气体识别）、B=Bearing（轴承 RUL）、C=Turbofan（涡扇发动机 RUL）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# SwanLab 的 group = 数据类型名，同组实验在页面上并排比较
DATASET_GROUPS = {"A": "Gas", "B": "Bearing", "C": "Turbofan"}


def dataset_group(profile: str) -> str:
    return DATASET_GROUPS.get(str(profile).upper(), "Other")


def load_swanlab_config(root: Path) -> dict:
    path = Path(root) / "configs" / "platform.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("swanlab", {})
    except Exception:
        return {}


class SwanLabReporter:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.cfg = load_swanlab_config(self.root)
        self.swanlab = None
        try:
            import swanlab  # noqa: F401
            self.swanlab = swanlab
        except Exception:
            self.swanlab = None
        self._run = None

    @property
    def available(self) -> bool:
        return self.swanlab is not None

    def start(self, profile: str, round_no: int, *, config: dict | None = None) -> None:
        if not self.swanlab:
            return
        group = dataset_group(profile)
        body = dict(config or {})
        body.setdefault("dataset_group", group)
        kwargs = {
            "project": self.cfg.get("project", "zhixun-engine"),
            "experiment_name": f"{profile}-round{round_no}",
            "group": group,
            "config": body,
        }
        if self.cfg.get("workspace"):
            kwargs["workspace"] = self.cfg["workspace"]
        if self.cfg.get("public"):
            kwargs["public"] = True
        if os.environ.get("SWANLAB_MODE") == "local":
            kwargs["mode"] = "local"
        if os.environ.get("SWANLAB_LOGDIR"):
            kwargs["logdir"] = os.environ["SWANLAB_LOGDIR"]
        try:
            self._run = self.swanlab.init(**kwargs)
        except TypeError:
            # 老版本 SDK 不认 group：退回不带分组，至少不丢记录
            kwargs.pop("group", None)
            try:
                self._run = self.swanlab.init(**kwargs)
            except Exception:
                self._run = None
        except Exception:
            self._run = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if not self._run:
            return
        try:
            self.swanlab.log(metrics, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if self._run:
            try:
                self.swanlab.finish()
            except Exception:
                pass
            self._run = None
