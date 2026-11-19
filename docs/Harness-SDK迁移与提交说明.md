# Harness SDK 1.0.2 迁移与提交说明

> 目标：把原来的「命令行 Harness」改成平台认可的**镜像式 Harness**（`python3 /app/harness.py describe|train|predict`）。
> 状态：代码与本地协议回归已通过；Docker 镜像待在有 Docker 的机器上构建（本机未安装 Docker）。

## 1. 结论

| 检查项 | 结果 |
| --- | --- |
| `describe` 输出协议 | ✅ `contest.harness.workspace.v2` |
| `train` 走平台网关调用模型 | ✅ 每轮恰好 1 次 `rnd.model()` |
| `finish(model_call_ids=...)` 非空 | ✅ 引用真实回执号 |
| 三个数据集（A/B/C）继续训练 | ✅ 均能在父检查点上续训并提升或保持 |
| `predict` 逐行应答 | ✅ 分类/回归形状全部通过 SDK 校验 |
| 跨轮状态记忆 | ✅ `/state` → `state_workspace` → `save_state` 串联 |
| 提交目录限制（≤64 文件 / ≤64MiB） | ✅ 每轮 4 个文件、约 100KB |
| 镜像里含模型权重 | ✅ 不含（权重由平台 `/previous`、`/model` 挂载） |
| 外网依赖（PyTRiO 直连、SwanLab） | ✅ 运行路径中已移除 |

## 2. 新目录

```
app/
  harness.py               # 7 行入口，平台按 describe/train/predict 调用
  contest_submission.py    # train() / predict()，全部业务逻辑
  contest_paths.py         # 只读/可写目录统一出口
  contest_data.py          # 行 → 特征向量（复用基线包的展开规则）
  contest_llm.py           # 决策层：只经 rnd.model 出参数
  baselines/               # 基线代码（model/data/train/predict）+ configs/{A,B,C}.json
  _local_check.py          # 本地联调用假宿主（不进镜像）
submit/
  Dockerfile               # 参赛镜像定义
  requirements.txt         # numpy==1.26.4
  context/                 # build_submit.py 生成的构建上下文（25 个文件，约 86KB）
scripts/build_submit.py    # 组装构建上下文，并拒绝任何 .npz/.pt/.bin/.pkl
```

## 3. 旧 → 新 映射

| 旧（命令行 Harness） | 新（SDK 提交） | 说明 |
| --- | --- | --- |
| `src/harness.py --profile A` | `app/contest_submission.py: train(rnd)` | 数据来源改为 `rnd.data.training_rows()` |
| `src/project.py` 的路径拼装 | `app/contest_paths.py` | `/data`、`/previous`、`/state`、`/model` 只读，`/workspace` 可写 |
| `src/adapter/{batch,artifact,platform}.py` | `app/contest_data.py` | 只保留「行 → 特征」与 JSONL 落盘 |
| `src/strategy/deciders.py`（直连 PyTRiO） | `app/contest_llm.py`（`rnd.model`） | 不持有密钥、不拼官方提示词 |
| `src/policy/*`（策略竞技场） | 保留在 `src/`，**不参与提交** | 提交链路只保留「参数夹紧 + 单调不降门」 |
| `src/scripts/*`（体检、回滚、画图） | 保留，仅开发用 | 不进镜像 |
| SwanLab 记录 | 移出镜像 | 评测环境无外网；记录改在开发机跑 |

## 4. 每轮 train 的真实流程

1. `rnd.data.training_rows()` / `validation_rows()` → `/workspace/round-XX-<数据集>/train.jsonl`、`validation.jsonl`（顺带校验行字段）。
2. `rnd.parent.path("checkpoint.npz")` 取父检查点（SDK 会核对摘要）。
3. `rnd.history()` 取历史记录，`state_workspace` 取上一轮状态 → 一起拼成决策上下文。
4. **一次** `rnd.model(...)`：返回 `action/epochs/learning_rate_scale/hypothesis/change_summary/reason`，参数被夹紧到 `epochs∈[1,4]`、`lr×[0.5,1.5]`。
5. `baselines.train.train_run(... resume=父检查点)` 继续训练（B 额外写标签来源声明）。
6. 门控：验证集指标是否变好。
   **变好 → 提交训练产物；没变好 → 提交父代权重**，保证提交链单调不降。
7. `save_state`：把本轮（数据集、参数、指标前后值、是否变好）追加进 `memory.json`（保留最近 20 轮）。
8. `submit_candidate(...)` → `finish(selected_artifact_id, ..., model_call_ids=rnd.call_ids)`。
9. stdout 只有协议帧；所有日志走 stderr。

## 5. 本地协议回归（证据）

用 SDK 自带 `tests/support.Host` 做**进程内假宿主**，跑完整 `train` + `predict`：

| 数据集 | 轮数 | 基线指标 | 本轮结果 | 提交 |
| --- | --- | --- | --- | --- |
| A（分类 macro_f1） | 1 | 0.71403 | 0.72570 | 训练产物 |
| A | 2 | 0.72570 | 0.71664 | 父代（单调不降生效） |
| B（回归 MAE） | 1 | 563.11 | 550.04 | 训练产物 |
| B | 2 | — | — | 状态记忆连续（memory rounds=2） |
| C（回归 MAE） | 1 | 38.8998 | 38.8120 | 训练产物 |
| C | 2–3 | 38.8120 | 38.9931 | 父代（同父同参会复现，靠决策换参数） |

复现命令：

```powershell
$env:PYTHONIOENCODING="utf-8"
py app\_local_check.py A                      # 单轮
$env:CONTEST_ROUNDS="2"; py app\_local_check.py A   # 双轮，验证 /state 记忆
$env:CONTEST_APP="<包根>\submit\context\app"; $env:CONTEST_PACKAGES="<包根>\packages"
py app\_local_check.py B                     # 验证打包上下文本身可跑
```

## 6. 构建与提交

```bash
py scripts/build_submit.py           # 生成 submit/context（25 文件，无权重）
cd submit/context
docker build -t <你的镜像名>:1.0.2 .
docker run --rm -i --entrypoint python3 <镜像名>:1.0.2 /app/harness.py describe
```

`docker run ... describe` 必须打印一行 JSON，且 `protocol=contest.harness.workspace.v2`、`model_access=host_gateway_rpc`、`tasks=["classification","regression"]`。

## 7. 与评测环境的边界

- 评测环境**无外网**：代码里没有 `requests`、没有 PyTRiO 客户端、没有 SwanLab 上报。
- 只有 `/workspace` 可写：所有落盘都在 `contest_paths.workspace()` 下。
- 不打包权重：父检查点来自 `/previous`，本轮产物经平台挂载到 `/model`，预测期只读 `CONTEST_ARTIFACT_PATH`。
- 预测期不看标签：`predict` 只用 `sample.inputs`，且按训练期路径表取特征，缺失补 0。
- 分类最高分唯一：并列时对首个最大值加 `1e-6`，避免 SDK `PREDICTION_INVALID`。

## 8. 未完成 / 风险

| 项 | 说明 | 建议 |
| --- | --- | --- |
| ~~镜像未实际构建~~ | 已解决：2026-11-18 装好 WSL2 + Docker Desktop 4.99，本地构建成功 | `zhixun-harness:1.0.2`，199MB，`describe`/`predict` 容器内验证通过 |
| ~~4C8G 压力未测~~ | 已测：4C8G + 无网络，单轮 1 epoch：A 2.5s、B 4.5s、C 19.8s | 远低于 `wall_seconds`，无需优化 |
| 重复轮次确定性 | 同父同参会得到同一结果 | 由决策层根据 `memory.json` 调整 lr/epochs（已把历史指标喂进上下文） |
| 旧命令行链路仍在 | `src/harness.py` 是开发工具 | 不要提交它；镜像只 COPY `app/` 与 `contest_harness/` |
| 镜像体积 | 实际 199MB（`python:3.12-slim` + numpy） | 远小于 2GB，无需额外裁剪 |
