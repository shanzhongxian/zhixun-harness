# 版本记录

| 版本 | 对应分支 | 日期 | 说明 |
|---|---|---|---|
| **v1** | `v1` | 2026-11-18 | 初版：Harness SDK 1.0.2 迁移完成；可跑 describe/train/predict；训练报告沿用随包基线格式 |
| **v2** | `v2` | 2026-11-21 | 修复版：强制从父检查点续训；训练报告改为自有格式并与产物摘要一致；拒绝路径不再复用训练报告 |
| **v3** | `v3` | 2026-11-23 | 十轮不再交父副本：候选恒为本轮真实训练产物；本轮最佳 epoch 择优；夹具指标否决；连续被拒强制换方向 |
| **v4** | `v4`、`main` | 2026-11-25 | 效率版：数据解析提速 6 倍；三个任务各一种效率方法（A 轮次级早停、B 哈希抽样、C 尾部保留分层抽样）；阶段首轮放宽容差；末端 cooldown |

## v2 相对 v1 的改动（2 个文件，+53 / −14 行）

| 文件 | 改动 |
|---|---|
| `app/baselines/train.py` | `resume` 改为必填；父检查点缺失直接报错；删除从头建模型分支；CLI 强制续训 |
| `app/contest_submission.py` | 训练报告改为 `contest.training-report.v1`（父/候选摘要、`initial_steps/final_steps`、`kept_parent`）；拒绝时不再复制训练报告 |

## v3 相对 v2 的改动（3 个文件，+280 / −57 行）

| 文件 | 改动 |
|---|---|
| `app/baselines/train.py` | 保存"本轮最佳 epoch"；`primary_metric` 决定择优口径；`deadline` 软截止；报告增加 `selected_epoch` / `validation_by_epoch` |
| `app/contest_submission.py` | 删除"未改善即复制父检查点"；候选恒为真实训练权重，采用与否由 `finish(selected_artifact_id=)` 决定；`no_op` 强制为 `train`；新增 `_guard` / `_rejections` / `_deadline` / `_clip` / `_recon`；新增 `probe.json` |
| `app/contest_llm.py` | 提示词去掉 `no_op`、加入拒绝次数与上轮参数；新增 `_diversify` 退火 |

## v4 相对 v3 的改动

| 文件 | 改动 |
|---|---|
| `app/baselines/data.py` | **解析提速**：首行建立"取值计划"（叶子访问路径），后续行按计划直接取值，不再为每个特征拼 `json.dumps`；每 1000 行做一次完整结构校验。C 的 113,089 行解析从 **12.9 秒降到 2.15 秒** |
| `app/baselines/efficiency.py`（新增） | 三个任务各一种效率方法，机制不同、均通用：A 轮次级早停、B 哈希抽样 70%、C 尾部保留分层抽样 35% |
| `app/baselines/train.py` | A 的早停判定；末端 cooldown（`epochs ≥ 2` 时最后一个 epoch 学习率降为 0.3 倍，checkpoint 内仍保存基准学习率）；报告新增 `efficiency` 块（策略、行数、每 epoch 秒数） |
| `app/contest_submission.py` | 效率策略在**物化阶段**生效（少写、少解析、少训练）；stderr 增加 `[timing]` 打点；阶段首轮采纳容差从 0 放宽到 0.5%；`probe.json` 记录效率统计 |
| `scripts/build_submit.py` | 构建上下文补上 `efficiency.py`（v4 首次构建时漏拷，镜像里缺文件导致 `PARTICIPANT_EXCEPTION`，已修复） |

### v4 实测（本机 4C8G 容器、`--network none`、每任务 2 轮）

| 任务 | v3 单轮 | v4 单轮 | 提速 |
|---|---|---|---|
| A | 1.8–2.3 s | 1.5–2.8 s | 无变化（A 只有 445 行，实测无重复、最近邻最小距离 0.30，数据层无可压缩） |
| C | 18.1–18.2 s | **7.6–7.7 s** | **2.4 倍** |

容器内 C 的物化仍占 5.3 秒（写 39,581 行到 `/workspace` 的 bind mount），是下一步的优化点。

## 镜像对应

| 版本 | 镜像 | 说明 |
|---|---|---|
| v4 | `zhixun-harness:v4` | 当前版本，镜像 ID `b1f21e026378`，199 MB |
| v3 | `zhixun-harness:v3` | 十轮真实续训版 |
| v2 | `zhixun-harness:1.0.3` | 已弃用 |
| v1 | `zhixun-harness:1.0.2` | 已弃用 |
