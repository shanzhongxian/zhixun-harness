"""本地联调：用 SDK 自带的 Host 假主机跑 train（可连续多轮），最后跑一次 predict。

用法：
  python app/_local_check.py [A|B|C]
  CONTEST_ROUNDS=2 python app/_local_check.py A                  # 验证跨轮状态记忆
  CONTEST_DOCKER=1 python app/_local_check.py A                  # 在容器里跑并记录耗时
  CONTEST_DOCKER=1 CONTEST_CPUS=4 CONTEST_MEM=8g python app/_local_check.py B
  CONTEST_APP=<submit/context/app> python app/_local_check.py A  # 验证打包上下文
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = Path(os.environ.get("CONTEST_APP") or Path(__file__).resolve().parent).resolve()
ROOT = APP.parent
SDK = Path(os.environ.get("CONTEST_SDK") or r"D:\比赛数据\_sdk102\contest-harness-sdk-1.0.2")
PACKAGE_ROOT = Path(os.environ.get("CONTEST_PACKAGES") or (ROOT / "packages"))
PACKAGES = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}
DOCKER = os.environ.get("CONTEST_DOCKER") == "1"
IMAGE = os.environ.get("CONTEST_IMAGE", "zhixun-harness:1.0.2")
CPUS = os.environ.get("CONTEST_CPUS", "4")
MEM = os.environ.get("CONTEST_MEM", "8g")
sys.path.insert(0, str(SDK))
from tests.support import Host  # noqa: E402
from tests.test_sdk import start_frame  # noqa: E402

PROMPT = "从官方基线与公开训练、验证数据出发，在授权预算内自主提出假设。"
DECISION = {"action": "train", "epochs": 1, "learning_rate_scale": 1.0,
            "hypothesis": "公开数据分布未漂移，做一轮短程继续训练即可。",
            "change_summary": "epochs=1，学习率保持基线值。", "reason": "验证集与训练集同源。"}


def fake_model(messages, options):
    assert messages[0]["role"] == "system", messages
    return {"request_id": "call-local-1", "id": "chatcmpl-local",
            "choices": [{"message": {"role": "assistant", "content": json.dumps(DECISION, ensure_ascii=False)}}]}


def _sha256(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rebind(frame, checkpoint, key):
    digest = _sha256(checkpoint)
    frame["parent_checkpoint_sha256"] = digest
    frame["parent_artifact"].update({"artifact_id": "baseline-%s" % key, "artifact_sha256": digest,
                                     "files": [{"path": "checkpoint.npz", "sha256": digest,
                                                "size_bytes": checkpoint.stat().st_size}]})
    frame["artifacts"] = [frame["parent_artifact"]]


def _train_command(tmp):
    if not DOCKER:
        return [sys.executable, str(APP / "harness.py"), "train"]
    return [
        "docker", "run", "--rm", "-i", "--network", "none",
        "--cpus", CPUS, "--memory", MEM,
        "-v", "%s:/data:ro" % (tmp / "data"),
        "-v", "%s:/previous:ro" % (tmp / "previous"),
        "-v", "%s:/workspace" % (tmp / "workspace"),
        IMAGE, "train",
    ]


def _predict_command(artifact):
    if not DOCKER:
        return [sys.executable, str(APP / "harness.py"), "predict"]
    return [
        "docker", "run", "--rm", "-i", "--network", "none",
        "--cpus", CPUS, "--memory", MEM,
        "-v", "%s:/model:ro" % artifact.parent,
        "-e", "CONTEST_ARTIFACT_PATH=/model/%s" % artifact.name,
        IMAGE, "predict",
    ]


def main(key):
    package = PACKAGE_ROOT / PACKAGES[key]
    tmp = Path(tempfile.mkdtemp(prefix="sdk-check-"))
    (tmp / "previous").mkdir()
    (tmp / "workspace").mkdir()
    (tmp / "model").mkdir()
    checkpoint = tmp / "previous" / "checkpoint.npz"
    shutil.copyfile(package / "checkpoint.npz", checkpoint)

    task = "classification" if key == "A" else "regression"
    if DOCKER:
        (tmp / "data").mkdir()
        for name in ("train.jsonl", "validation.jsonl"):
            shutil.copyfile(package / "public" / name, tmp / "data" / name)
        train_path, validation_path = "/data/train.jsonl", "/data/validation.jsonl"
    else:
        train_path, validation_path = package / "public" / "train.jsonl", package / "public" / "validation.jsonl"

    frame = start_frame(train_path, validation_path)
    frame.update({
        "dataset_key": key, "prompt": PROMPT,
        "task_contract": {**frame["task_contract"], "dataset_key": key, "task": task,
                          "metric": "macro_f1" if task == "classification" else "mae",
                          "metric_direction": "higher" if task == "classification" else "lower"},
    })
    _rebind(frame, checkpoint, key)

    env = os.environ.copy()
    search = [str(APP)]
    if (APP.parent / "contest_harness").is_dir():
        search.append(str(APP.parent))
    search.append(str(SDK))
    env["PYTHONPATH"] = os.pathsep.join(search)
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(tmp / "workspace")
    env["CONTEST_LOCAL_ROOT"] = str(tmp)
    env["PYTHONIOENCODING"] = "utf-8"

    rounds = int(os.environ.get("CONTEST_ROUNDS", "1"))
    state_dir = None
    work = tmp / "workspace"
    timings = []
    for index in range(1, rounds + 1):
        if index > 1:
            latest = sorted(work.glob("round-*/candidate/checkpoint.npz"))[-1]
            shutil.copyfile(latest, checkpoint)
            _rebind(frame, checkpoint, key)
        frame["round_index"] = index
        if state_dir is None:
            frame["state_path"] = None
        elif DOCKER:
            frame["state_path"] = "/workspace/" + state_dir.relative_to(work).as_posix()
        else:
            frame["state_path"] = str(state_dir)
        started = time.perf_counter()
        process = subprocess.Popen(_train_command(tmp), cwd=str(APP),
                                   env=None if DOCKER else env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace")
        host = Host(frame, fake_model, history=[{"id": "round-prev", "round_index": 0, "dataset_key": key}])
        thread = host.serve(process)
        stderr = process.stderr.read()
        code = process.wait(timeout=3600)
        elapsed = time.perf_counter() - started
        thread.join(timeout=5)
        timings.append(elapsed)
        print("[round %d] train exit=%s failure=%s wall=%.1fs" % (index, code, host.failure, elapsed))
        if stderr:
            print("--- stderr ---\n" + stderr.strip())
        if code != 0 or host.failure:
            return 1
        print("record:", json.dumps(host.record, ensure_ascii=False)[:600])
        state_dir = sorted(work.glob("round-%02d-*/state" % index))[-1]
        memory = json.loads((state_dir / "memory.json").read_text(encoding="utf-8"))
        print("memory rounds:", len(memory["rounds"]))

    candidates = sorted(work.glob("round-*/candidate/checkpoint.npz"))
    assert candidates, list(work.rglob("*"))
    artifact = candidates[-1]
    rows = [json.loads(line) for line in (package / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:6]
    payload = "\n".join(json.dumps({"task": task, "inputs": row["inputs"]}, ensure_ascii=False) for row in rows) + "\n"
    started = time.perf_counter()
    done = subprocess.run(_predict_command(artifact), cwd=str(APP), env=None if DOCKER else dict(
        env, CONTEST_ARTIFACT_PATH=str(artifact), CONTEST_DATASET_KEY=key),
        input=payload, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    print("predict exit=%s wall=%.1fs" % (done.returncode, time.perf_counter() - started))
    if done.returncode:
        print(done.stderr)
        return 1
    for line in done.stdout.strip().splitlines():
        print("  out:", line)
    print("train wall:", ", ".join("%.1fs" % value for value in timings))
    print("worktree:", [p.relative_to(work).as_posix() for p in sorted(work.rglob('*')) if p.is_file()][:24])
    print("tmp:", tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "A"))
