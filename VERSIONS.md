# 版本记录

| 版本 | 对应分支 | 日期 | 说明 |
|---|---|---|---|
| **v1** | `v1` | 2026-11-18 | 初版：Harness SDK 1.0.2 迁移完成；可跑 describe/train/predict；训练报告沿用随包基线格式 |
| **v2** | `v2`、`main` | 2026-11-21 | 修复版：强制从父检查点续训；训练报告改为自有格式并与产物摘要一致；拒绝路径不再复用训练报告 |
| **v3** | `v3` | 2026-11-23 | 十轮不再交父副本：候选恒为本轮真实训练产物，采用与否由选定产物决定；本轮最佳 epoch 择优；夹具指标否决；连续被拒强制换方向；每轮曲线与差距落盘 |

## v2 相对 v1 的改动（2 个文件，+53 / −14 行）

| 文件 | 改动 |
|---|---|
| `app/baselines/train.py` | `resume` 改为必填；父检查点缺失直接报错；删除从头建模型分支；CLI 强制续训 |
| `app/contest_submission.py` | 训练报告改为 `contest.training-report.v1`（父/候选摘要、`initial_steps/final_steps`、`kept_parent`）；拒绝时不再复制训练报告；`decision.json` 增加 `parent_artifact_id`、`resumed` |

## v3 相对 v2 的改动（3 个文件，+280 / −57 行）

| 文件 | 改动 |
|---|---|
| `app/baselines/train.py` | 保存"本轮最佳 epoch"而不是最后一个 epoch（`copy.deepcopy` 快照 + 按 `primary_metric` 择优）；新增 `deadline` 软截止，只在已开始训练后生效且至少完成 1 个 epoch；报告新增 `selected_epoch`、`epochs_completed`、`epoch_stop_reason`、`validation_by_epoch`、`loss_by_epoch` |
| `app/contest_submission.py` | 删除 `keep_trained → copyfile(parent)`：候选恒为本轮真实训练权重；`finish(selected_artifact_id=)` 在未改善时选回 `rnd.parent.artifact_id`（SDK 文档明确允许"保留父模型"）；`no_op` 强制改为 `train`；新增夹具指标否决 `_guard`、连续未采纳计数 `_rejections`、阶段感知 `stage_changed`、墙钟软截止 `_deadline`、UTF-8 字节级截断 `_clip`、stderr 侦察日志 `_recon`；新增 `probe.json`；`training.json` 增加 `selected_checkpoint_sha256`、`trained_from_parent`、`vetoed`、`secondary` |
| `app/contest_llm.py` | 提示词去掉 `no_op`、加入拒绝次数与上一轮实际参数；新增 `_diversify`：连续 2 轮被拒且参数与上一轮重复时强制改档（`learning_rate_scale` 1.5↔0.5、`epochs` 4↔2），并在 `reason`/`change_summary` 写明原因 |

### v3 的设计约束

- 平台要求始终基于父检查点续训，因此本轮不存在"不训练"的分支：`resume=parent` 强制，父缺失立即报错。
- 候选目录里的 `checkpoint.npz` 一定来自本轮训练；是否采用不体现在产物上，而体现在 `finish` 的 `selected_artifact_id`。
- 报告里同时写 `candidate_checkpoint_sha256`（提交的产物）与 `selected_checkpoint_sha256`（本轮选定），两者不一致时必然对应"选定父产物"，不会产生歧义。
- `reason`、`hypothesis`、`change_summary` 一律按 UTF-8 **字节**截断（SDK 上限 4000 字节，不是字符）。

## 镜像对应

| 版本 | 镜像 | tar SHA256 | 说明 |
|---|---|---|---|
| v3 | `zhixun-harness:v3` | `C7CDE11CD74896D38E8B584D733896004DED507ECDF0C82E3DF4A9543927F3B9` | 当前提交版本，195.12 MB |
| v2 | `zhixun-harness:1.0.3` | `816DA84632F6FCC4D4FC35191F0A03B63D2CDAF1EEBB55549306DACCB89C13B8` | 已弃用 |
| v1 | `zhixun-harness:1.0.2` | `FE892F15…` | 已弃用 |
