# 智训引擎 · Harness 自演化训练系统

工业模型（NumPy MLP）在工况漂移下的自演化训练闭环：诊断 → 训练师决策 → 续训 → 门禁 → 版本。

- 训练师：Qwen3.8-27B（本地经 PyTRiO，评测经平台网关 `rnd.model`），只做决策，不训练模型
- 工业模型：三工况 MLP（气体识别 4326 参数 / 轴承 RUL 865 / 涡扇 RUL 833），纯 CPU
- 交付：符合 Harness SDK 1.0.2 的容器镜像（`/app/harness.py describe|train|predict`）

## 版本

| 版本 | 分支 | 说明 |
|---|---|---|
| **v4** | `v4`（= `main`） | 当前版本：数据解析提速 6 倍；A/B/C 各配一种效率方法（轮次早停 / 哈希抽样 / 尾部保留分层抽样）；阶段首轮放宽容差；末端 cooldown |
| **v3** | `v3` | 候选恒为本轮真实训练产物（不再交父副本），本轮最佳 epoch 择优，夹具指标否决与"连续被拒强制换方向" |
| **v2** | `v2` | 强制从父检查点续训，训练报告与产物摘要一致 |
| **v1** | `v1` | 初版：SDK 迁移完成，但训练报告沿用随包格式、代码保留从头训练分支 |

详见 [VERSIONS.md](VERSIONS.md)。

## 目录

```
app/            提交入口：harness、contest_submission、路径/数据/决策
  baselines/    基线模型、训练、预测与三工况配置
src/            本地开发链路：harness、策略层、脚本、单元测试
scripts/        构建上下文组装（拒绝权重文件）
configs/        平台协议参数
policies/       冠军策略
docs/           使用说明、平台协议、审阅报告、SDK 迁移说明等
packages/       官方三个训练包（仅 baselines 代码与配置，不含数据与权重）
```

## 快速开始

```bash
py scripts/build_submit.py
cd submit/context
docker build -t zhixun-harness:1.0.2 .
docker run --rm -i --entrypoint python3 zhixun-harness:1.0.2 /app/harness.py describe
```

本地联调（假宿主）：

```bash
py app/_local_check.py A
$env:CONTEST_ROUNDS="2"; py app/_local_check.py A
$env:CONTEST_DOCKER="1"; py app/_local_check.py A
```

## 结果（公开验证集）

| 数据集 | 静态基线 | 当前最优 |
|---|---|---|
| A · macro_f1 ↑ | 0.714030 | 0.719784 |
| B · rmse ↓ | 681.159833 | 651.626661 |
| C · rmse ↓ | 52.868573 | 51.845074 |

## 实验记录

SwanLab（公开）：https://swanlab.cn/@shanzhongxian/zhixun-engine/v1/dd0idn/overview

## 说明

- 仓库不含模型权重、不含 API Key；`packages/` 仅保留 baselines 代码与配置。
- 镜像与可复现材料见交付包（tar 体积 195MB，超出 GitHub 单文件限制，走 Release 或网盘）。
