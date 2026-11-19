"""组装可提交的镜像构建上下文（submit/context）。

只拷贝运行时真正需要的文件：app 的 5 个模块 + baselines 代码与配置 + SDK。
不拷贝权重、不拷贝 _local_check.py、不拷贝任何密钥或日志。

用法：python scripts/build_submit.py [--sdk D:\\path\\to\\contest-harness-sdk-1.0.2]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SUBMIT = ROOT / "submit"
CONTEXT = SUBMIT / "context"
MODULES = ["harness.py", "contest_submission.py", "contest_paths.py", "contest_data.py", "contest_llm.py"]
BASELINES = ["__init__.py", "data.py", "model.py", "predict.py", "train.py"]
CONFIGS = ["A.json", "B.json", "C.json"]


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise SystemExit("缺少文件：%s" % source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", default=r"D:\比赛数据\_sdk102\contest-harness-sdk-1.0.2")
    args = parser.parse_args(argv)
    sdk = Path(args.sdk)
    package = sdk / "contest_harness"
    if not (package / "cli.py").is_file():
        raise SystemExit("SDK 目录不像 contest_harness 包：%s" % sdk)

    if CONTEXT.exists():
        shutil.rmtree(CONTEXT)
    CONTEXT.mkdir(parents=True)

    for name in MODULES:
        copy_file(APP / name, CONTEXT / "app" / name)
    for name in BASELINES:
        copy_file(APP / "baselines" / name, CONTEXT / "app" / "baselines" / name)
    for name in CONFIGS:
        copy_file(APP / "baselines" / "configs" / name, CONTEXT / "app" / "baselines" / "configs" / name)
    for item in sorted(package.glob("*.py")):
        copy_file(item, CONTEXT / "contest_harness" / item.name)

    copy_file(SUBMIT / "Dockerfile", CONTEXT / "Dockerfile")
    copy_file(SUBMIT / "requirements.txt", CONTEXT / "requirements.txt")
    copy_file(SUBMIT / "THIRD_PARTY.md", CONTEXT / "THIRD_PARTY.md")

    files = sorted(path.relative_to(CONTEXT).as_posix() for path in CONTEXT.rglob("*") if path.is_file())
    total = sum((CONTEXT / name).stat().st_size for name in files)
    print("context: %s" % CONTEXT)
    print("files: %d, bytes: %d" % (len(files), total))
    for name in files:
        print("  " + name)
    weights = [name for name in files if name.endswith((".npz", ".pt", ".bin", ".pkl"))]
    if weights:
        raise SystemExit("构建上下文里出现了权重文件：%s" % weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
