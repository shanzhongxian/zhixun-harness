# 参赛镜像构建目录

按 Harness SDK 1.0.2 协议提交：平台执行 `python3 /app/harness.py describe|train|predict`。

## 一键构建

```bash
# 1) 在包根目录生成构建上下文（会自动拒绝 .npz/.pt/.bin/.pkl）
py scripts/build_submit.py

# 2) 构建镜像
cd submit/context
docker build -t zhixun-harness:1.0.2 .

# 3) 协议冒烟：必须打印一行 JSON
docker run --rm -i --entrypoint python3 zhixun-harness:1.0.2 /app/harness.py describe

# 4) 容器内完整 train + 计时（4C8G、无网络）
$env:CONTEST_DOCKER="1"; py app\_local_check.py A   # A≈2.5s / B≈4.5s / C≈19.8s
# {"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc",
#  "sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}
```

## 上下文内容（25 个文件，约 86 KB）

| 路径 | 作用 |
| --- | --- |
| `app/harness.py` | 平台入口，7 行，转交 SDK CLI |
| `app/contest_submission.py` | `train(rnd)` / `predict(sample, model_dir)` |
| `app/contest_paths.py` | `/data`、`/previous`、`/state`、`/model`、`/workspace` |
| `app/contest_data.py` | 行 → 特征向量，复用基线展开规则 |
| `app/contest_llm.py` | 决策层，只经 `rnd.model()`，不持密钥 |
| `app/baselines/*.py` | 基线模型/训练/预测代码 |
| `app/baselines/configs/{A,B,C}.json` | 三工况配置（B 是 regression） |
| `contest_harness/*.py` | 官方 SDK 包（随镜像安装，免 pip 外网） |
| `Dockerfile`、`requirements.txt` | `python:3.12-slim` + `numpy==1.26.4` |
| `THIRD_PARTY.md` | 第三方组件与许可声明（随镜像进 `/app`） |

## 镜像约束

- 不含任何模型权重：父检查点来自 `/previous`，本轮产物由平台挂载到 `/model`。
- 不含密钥：无 PyTRiO key、无 SwanLab key、无 `.swanlab`。
- 无外网调用：不使用 `requests`/`pytrio`/`swanlab`，模型调用只走 `rnd.model()`。
- 只写 `/workspace`：所有落盘经 `contest_paths.workspace()`；镜像内预建 `/workspace` 并 chown 给 65534，挂载只读时自动回退 `/tmp/workspace`。
- 以 `USER 65534:65534` 运行。

## 注意

- 镜像已在本机构建并验证（Docker Desktop 4.99 + WSL2）：199MB，`describe`/`predict`/断网冒烟均通过。
- 构建上下文由脚本生成，**不要手工改 `context/`**，改 `app/` 后重新跑 `build_submit.py`。
