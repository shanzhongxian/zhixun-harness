# 第三方组件与许可声明

本镜像只包含运行 Harness 所需的最小代码与依赖，不含任何模型权重、密钥或外部数据。

## 运行环境

| 组件 | 版本 | 许可 | 用途 |
|---|---|---|---|
| Python | 3.12（`python:3.12-slim-bookworm` 官方镜像） | PSF License | 运行时 |
| NumPy | 1.26.4 | BSD 3-Clause | 工业模型（MLP）推理与训练 |

## 随镜像分发的代码

| 组件 | 来源 | 说明 |
|---|---|---|
| `contest_harness` | 赛事官方 Harness SDK 1.0.2 | 按赛题要求随镜像安装，未修改 |
| `app/baselines/*.py` | 赛事随包基线代码 | 仅做续训/预测所需的裁剪，算法未变 |
| `app/baselines/configs/{A,B,C}.json` | 赛事随包基线配置 | B 为 regression 配置，A/C 与随包一致 |
| `app/*.py` | 本队编写 | 协议适配、决策接入、路径与数据展开 |

## 未包含

- 任何 `.npz` / `.pt` / `.bin` 模型权重（父产物由平台挂载到 `/previous`，本轮产物由平台挂载到 `/model`）；
- 任何 API Key、SwanLab 凭据、`~/.pytrio` 配置；
- 任何外网调用客户端（PyTRiO SDK、requests、urllib 请求、SwanLab 上报）。

## 数据

训练与验证数据来自平台只读挂载的 `/data/train.jsonl`、`/data/validation.jsonl`，不随镜像分发。
