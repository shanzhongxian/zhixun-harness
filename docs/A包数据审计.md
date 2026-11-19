# A 包数据审计（T02 前置）

数据位置：`<HARNESS_DATA_ROOT>/XJTU-SY_Bearing_Datasets`（原始数据不在包内，用环境变量指向）

## 1. 结论先行

这个"a包合并包"是**原始振动数据**（XJTU-SY 轴承全寿命），不是可直接训练的 JSONL。
它对应的是 **B 工况（轴承 RUL）** 的上游原料，而不是 A 工况（气体识别）。
A/B/C 三个可直接训练的成品在 `cpu-*-20261110-v1/public/` 里。

因此：**批次适配层只认 JSONL**，原始 CSV 不进训练链路；它的价值是校验特征口径与
核对 B 包标签公式，不参与运行时读取。

## 2. 结构

```
XJTU-SY_Bearing_Datasets/
├── 35Hz12kN/     5 根轴承
├── 37.5Hz11kN/   5 根轴承
├── 40Hz10kN/     5 根轴承
└── Introduction_to_XJTU-SY_Bearing_Dataset.pdf
├── <工况>/<BearingX_Y>/<序号>.csv
```

| 项 | 实测值 |
|---|---|
| 轴承总数 | 15（3 工况 × 5 根） |
| CSV 文件数 | 9216 |
| 单 CSV 行数 | 32769（1 行表头 + 32768 采样点） |
| 单 CSV 列 | `Horizontal_vibration_signals,Vertical_vibration_signals` |
| 采样率 | 25.6 kHz（每文件 1.28 s） |
| 每根轴承快照数 | 最少 42，最多 2538（长尾极不均匀） |

每根轴承快照数（决定 RUL 标签上限，也决定 B 包样本分布）：

| 工况 | 轴承数 | 最少 | 最多 |
|---|---|---|---|
| 35Hz12kN | 5 | 52 | 161 |
| 37.5Hz11kN | 5 | 42 | 533 |
| 40Hz10kN | 5 | 114 | 2538 |

## 3. 与 B 包的关系

B 包 `TASK.md` 定义：

- 34 维特征 = 每通道 17 个（RMS、峰值、峭度、频谱质心、频带能量等），双通道拼接；
- 标签 `RUL_minutes = (N - i) × 1`，最后一条为 0；
- 训练/验证/隐藏**按整根轴承切分**，同一轴承不得跨集合。

原始 CSV → JSONL 的转换脚本（`scripts/prepare_xjtu_bearing.py`）**不在包内**，
所以不能从原始数据重建 B 包；它只能用来抽样验证特征量级是否合理。

## 4. 对代码的影响

1. `adapter/batch.py` 只接受 JSONL 批次（`id/inputs/answer/meta`），复用
   `project.read_dataset` 做 schema 校验；
2. 分组键固定用 `meta.bearing`，不引入文件名解析；
3. 长尾分布提醒：B 包验证集 RMSE 会被长寿命轴承主导，门禁必须看**分组指标**
   （`strategy/observability.py::group_metrics`），不能只看总体 RMSE。

## 5. 抽样核对建议（不阻塞 T02）

- 抽 3 根轴承（各工况 1 根），用现有 `baselines/data.py` 的特征函数复算 34 维，
  与 B 包同轴承样本比对量级；
- 核对 `N - i` 与 B 包 `answer` 的一致性（任取 5 条）。

以上两项属于 T04（EDA）范畴，本阶段只做记录，不执行。
