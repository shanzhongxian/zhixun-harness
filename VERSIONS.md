# 版本记录

| 版本 | 对应分支 | 日期 | 说明 |
|---|---|---|---|
| **v1** | `v1` | 2026-11-18 | 初版：Harness SDK 1.0.2 迁移完成；可跑 describe/train/predict；训练报告沿用随包基线格式 |
| **v2** | `v2`、`main` | 2026-11-21 | 修复版：强制从父检查点续训；训练报告改为自有格式并与产物摘要一致；拒绝路径不再复用训练报告 |

## v2 相对 v1 的改动（2 个文件，+53 / −14 行）

| 文件 | 改动 |
|---|---|
| `app/baselines/train.py` | `resume` 改为必填；父检查点缺失直接报错；删除从头建模型分支；CLI 强制续训 |
| `app/contest_submission.py` | 训练报告改为 `contest.training-report.v1`（父/候选摘要、`initial_steps/final_steps`、`kept_parent`）；拒绝时不再复制训练报告；`decision.json` 增加 `parent_artifact_id`、`resumed` |

## 镜像对应

| 版本 | 镜像 | 说明 |
|---|---|---|
| v2 | `zhixun-harness:1.0.3` | 当前提交版本 |
| v1 | `zhixun-harness:1.0.2` | 已弃用 |
