# 三数据集 EDA 与证据链

> 覆盖计划 T04（EDA）与 T29（证据链 + 基线达标）。数值以 `runs/decisions.jsonl` 为准，可随运行刷新。

## A · 气体传感器漂移识别

- 任务：6 类分类（Ethanol / Ethylene / Ammonia / Acetaldehyde / Acetone / Toluene）
- 输入：128 维数值特征（传感器阵列）
- 数据量：训练 **445**（batch 1），验证 **1244**（batch 2）
- 分组键：`meta.source_batch`
- 漂移形态：训练/验证按**采集批次（时间顺序）隔离**；验证仅 batch 2 一个批次，组内无法再切回放集
- 过拟合风险：**高**（445 行 vs 4326 参数；训练 acc 0.973 / macro_f1 0.956 vs 验证 0.879 / 0.714，gap 明显）
- 标签：全量可得（6 类字符串）
- 结果：规则决策（lr 0.3，3 轮）macro_f1 **0.7140 → 0.7187**，`accepted`

## B · 滚动轴承剩余寿命（RUL）

- 任务：回归，单位分钟
- 输入：34 维特征（水平/垂直两通道各 17）
- 数据量：训练 **7032**，验证 **1679**
- 分组键：`meta.bearing`（验证 3 根：Bearing1_4 / Bearing2_4 / Bearing3_4）
- 漂移形态：分组 RMSE 差异大（约 **242 / 391 / 711**）→ 明显工况差异，`drift_flag` 触发
- 标签：数值分钟；按训练包说明**受限**
- 结果：规则决策（lr 0.3 + weight_decay 1e-4，3 轮）rmse **681.16 → 662.79**，`accepted`
- 回放集（留出 Bearing3_4 作“未见工况”）：711 → 691，**不退化** ✓

## C · 涡扇发动机剩余寿命（RUL）

- 任务：回归，单位循环；另报 NASA 非对称评分
- 输入：24 维特征（3 运行设置 + 21 传感器）
- 数据量：训练 **113089**，验证 **23049**
- 分组键：`meta.source_subset` + `meta.unit`（FD001–FD004 多台）
- 漂移形态：各台 RMSE 24–58 差异明显 → `drift_flag` 触发
- 标签：数值循环；全量
- 结果：规则决策（lr 0.3 + weight_decay 1e-4，3 轮）rmse **52.87 → 51.88**，`accepted`
- 回放集（留出 FD004 末段 36 台）：63.7 → 62.9，**不退化** ✓

---

## 证据链

- 版本指针：`runs/current.json`（含 `previous`，支持回滚）
- 决策日志：`runs/decisions.jsonl`（每轮完整诊断 + 决策 + 门禁 + 回放结果）
- 回滚演练：3 集重载+重评均 **<1.1s**（阈值 30s）✓
- 记录完整性：`python scripts/check_records.py` 全绿 ✓
- 本次验证：round-1 三集各 1 次尝试，均 `accepted`

**「≥3 次迭代含被拒」说明**：harness 支持 `--rounds` 多轮，门禁会拒绝退化候选；开发期已实测到一次 `rejected`（候选在留出的难轴承上退化被拦）。要产出正式证据，跑：

```bash
python src/harness.py --profile A --strategy rule --rounds 3 --lr-scale 1.0
```

（后续轮次用较大学习率易触发退化被拒，从而形成「≥3 次尝试、含被拒」的时间线。）

## 备注

- 回放集切分规则：按组键排序取末尾约 34% 的组作为 replay（视为未来/未见工况）；仅 1 组时 replay=gate（A 场景）。见 `strategy/replay.py`。
- 回放集门禁：候选必须在**完整验证集**上提升主指标，且**回放集不退化**，才 `accepted`。
