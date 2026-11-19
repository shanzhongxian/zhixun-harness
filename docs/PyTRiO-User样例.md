# PyTRiO 控制台 User 输入样例（真实数据生成）

> 来源：`runs/decisions.jsonl` + `policies/p001.json`，非手工编造。用的时候整段粘进 User 输入框。
> 对应 System 提示词见 `docs/PyTRiO控制台调教.md`。

## A（分类，macro_f1，连拒 10 次——最该测“换方向”）

```json
{
  "dataset": "A",
  "task": "classification",
  "metric": "macro_f1",
  "metric_direction": "higher",
  "diagnostics": {
    "train_loss": 0.127982,
    "validation_loss": 0.527869,
    "gap": 0.399887,
    "train_metrics": {"accuracy": 0.973034, "macro_f1": 0.955558},
    "validation_metrics": {"accuracy": 0.88746, "macro_f1": 0.719784},
    "group_summary": {
      "n_groups": 1,
      "mean": 0.719784,
      "max": 0.719784,
      "min": 0.719784,
      "worst3": [{"group": [2], "primary": 0.719784}]
    },
    "overfit_flag": false,
    "drift_flag": false,
    "thresholds_used": {"gap": 0.6, "spread": 0.5}
  },
  "current_policy": {"lr_scale": 0.3, "epochs": 5, "patience": 2},
  "previous": [
    {"round": 11, "action": "train",
     "params": {"lr_scale": 0.3, "epochs": 2, "patience": 2, "weight_decay": 0.0},
     "before": 0.719784, "after": 0.719784, "accepted": false},
    {"round": 12, "action": "train",
     "params": {"lr_scale": 0.3, "epochs": 2, "patience": 2, "weight_decay": 0.0},
     "before": 0.719784, "after": 0.719784, "accepted": false}
  ],
  "rejections": 10
}
```

期望：`lr_scale` 不再是 0.3，或加 `dropout`/`weight_decay`，且 `epochs` 不要继续加。

## B（回归，rmse，过拟合 + 漂移，连拒 7 次）

```json
{
  "dataset": "B",
  "task": "regression",
  "metric": "rmse",
  "metric_direction": "lower",
  "diagnostics": {
    "train_loss": 0.088988,
    "validation_loss": 0.694348,
    "gap": 0.60536,
    "train_metrics": {"mae": 162.824309, "rmse": 230.08942},
    "validation_metrics": {"mae": 539.823027, "rmse": 651.626661},
    "group_summary": {
      "n_groups": 3,
      "mean": 437.346731,
      "max": 678.934474,
      "min": 274.399497,
      "worst3": [
        {"group": ["Bearing3_4"], "primary": 678.934474},
        {"group": ["Bearing2_4"], "primary": 358.706222},
        {"group": ["Bearing1_4"], "primary": 274.399497}
      ]
    },
    "overfit_flag": true,
    "drift_flag": true,
    "thresholds_used": {"gap": 0.6, "spread": 0.5}
  },
  "current_policy": {"lr_scale": 0.3, "epochs": 5, "patience": 2},
  "previous": [
    {"round": 8, "action": "train",
     "params": {"lr_scale": 0.15, "epochs": 2, "patience": 1, "weight_decay": 0.0},
     "before": 651.626661, "after": 651.626661, "accepted": false},
    {"round": 9, "action": "train",
     "params": {"lr_scale": 0.15, "epochs": 2, "patience": 1, "weight_decay": 0.0},
     "before": 651.626661, "after": 651.626661, "accepted": false}
  ],
  "rejections": 7
}
```

期望：加正则（dropout/weight_decay），或 `loss=huber` 改善稳定性，别只动 lr。

## C（回归，rmse，106 组，漂移，连拒 7 次）

```json
{
  "dataset": "C",
  "task": "regression",
  "metric": "rmse",
  "metric_direction": "lower",
  "diagnostics": {
    "train_loss": 0.428041,
    "validation_loss": 0.383439,
    "gap": -0.044602,
    "train_metrics": {"mae": 40.082194, "rmse": 54.904752},
    "validation_metrics": {"mae": 38.095514, "rmse": 51.845074},
    "group_summary": {
      "n_groups": 106,
      "mean": 43.934037,
      "max": 131.317891,
      "min": 22.589624,
      "worst3": [
        {"group": ["FD004", 179], "primary": 131.317891},
        {"group": ["FD004", 201], "primary": 105.354953},
        {"group": ["FD004", 184], "primary": 96.252885}
      ]
    },
    "overfit_flag": false,
    "drift_flag": true,
    "thresholds_used": {"gap": 0.6, "spread": 0.5}
  },
  "current_policy": {"lr_scale": 0.3, "epochs": 5, "patience": 2},
  "previous": [
    {"round": 8, "action": "train",
     "params": {"lr_scale": 0.3, "epochs": 2, "patience": 2, "weight_decay": 0.0001},
     "before": 51.845074, "after": 51.845074, "accepted": false},
    {"round": 9, "action": "train",
     "params": {"lr_scale": 0.3, "epochs": 2, "patience": 2, "weight_decay": 0.0001},
     "before": 51.845074, "after": 51.845074, "accepted": false}
  ],
  "rejections": 7
}
```

期望：识别出最差组集中在 FD004，给小步 + 正则，不要指望一轮翻盘。

## 字段来源对照

| 字段 | 来自 |
|---|---|
| `diagnostics.*` | `runs/decisions.jsonl` 里该轮记录的 `diagnostics` |
| `group_summary` | 由 `group_metrics` 聚合（原有 106 条明细，压成统计 + 最差 3 组） |
| `current_policy` | `policies/p001.json` 的 `search` |
| `previous` | 最近两轮的 `action/params/validation_metrics/candidate_metric/result` |
| `rejections` | 该轮记录的 `consecutive_rejections_before` |
