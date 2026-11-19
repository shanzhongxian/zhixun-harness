# PyTRiO 控制台调教手册（对着浏览器页面操作）

> 对应参数基线：`docs/平台协议.md` 6.1。代码侧已按同一套默认值实现（`src/pytrio_client.py`）。
> 目的：在控制台里把 Qwen/Qwen3.8-27B 调成「稳定输出合法决策 JSON」的训练师，再把结论搬回代码。

---

## 一、先把参数拧到基线

打开 PyTRiO 控制台（Qwen/Qwen3.8-27B 对话页），右侧「参数配置」逐项改：

| 控制台项 | 现在 | 改成 | 为什么 |
|---|---|---|---|
| Temperature | 0.7 | **0** | 决策要可复现；0.7 会让同一份诊断给出三种不同动作 |
| Top p | 0.95 | **1.0** | 温度 0 时不起作用，保持一致 |
| 推理 Token 限制 | 4096 | **1024** | 决策 JSON 只有 100–200 token，4096 只会让模型写长推理 |
| Enable thinking | 开 | **关** | 思考文本挤占 token，还容易把严格 JSON 冲坏 |
| Seed | 空 | **42** | 固定，同输入同输出 |
| Presence penalty | 0 | **0** | 决策不需要去重 |
| Frequency penalty | 0 | **0** | 同上 |
| 停止词 | 空 | **空** | 加 `}` 之类会把 JSON 截断 |

改完先做一次**复现性自检**：同一段输入连发 3 次，温度 0 应给出完全相同的参数；如果三次不同，说明温度没生效或 seed 没填。

---

## 二、「官方模版」怎么用

页面底部有「官方模版」快捷按钮：

1. 点它 → 把官方 prompt 贴进 System → 记下它的结构和大致的 token 长度（评测时宿主会把这段拼在你的 system 前面）；
2. **不要**把官方 prompt 抄进代码或再贴一遍——重复会被判协议错；
3. 清空对话，换上下面这套自写提示词。

---

## 三、System（复制即用）

```
你是工业模型的训练师。每轮只做一次决策，只输出一个 JSON 对象，不要输出任何解释、思考过程或 Markdown 代码块。

可选动作：
- train：用公开训练/验证数据继续训练当前检查点
- no_op：不训练，沿用父代
- rollback：回退到上一版检查点

参数边界：epochs 1..4；lr_scale 0.5..1.5（相对基线学习率）；patience 1..3；
weight_decay 0..0.001；可选 dropout 0..0.3。不得改变网络结构、不得引入外部数据。

输出 schema：
{"action":"train|no_op|rollback",
 "hypothesis":"一句话预测本轮会发生什么",
 "evidence_ids":["你引用的观测字段名"],
 "params":{"lr_scale":1.0,"epochs":1,"patience":2}}

决策规则：
1. 先判状态：train 指标远高于 validation、gap 大 → 过拟合；分组指标 spread 大或 drift_flag=true → 漂移；两者都无 → 稳定。
2. 稳定且上次被拒 → 换方向（动 lr 或 epochs），禁止重复上一轮同样的参数。
3. 连续 2 次被拒 → 步长减半；连续 3 轮主指标不动 → 允许 no_op 并说明理由。
4. 不确定时选保守：epochs=1、lr_scale=1.0。
5. evidence_ids 只填你真正看过的字段，不要编造。
```

### 两个示例（接在 System 末尾，一次给够）

```
示例1（稳定场景，输入略）：
{"action":"train","hypothesis":"验证与训练同步，小步续训可提升。","evidence_ids":["gap","validation_metrics"],"params":{"lr_scale":1.0,"epochs":1,"patience":2}}

示例2（连续被拒场景，输入略）：
{"action":"train","hypothesis":"同参两次无效，改为减半步长并延长耐心。","evidence_ids":["rejections","gap"],"params":{"lr_scale":0.5,"epochs":2,"patience":3}}
```

---

## 四、User：三个测试场景逐个发

### 场景 A-1 稳定（应输出小步 train）

```json
{"dataset":"A","task":"classification","metric":"macro_f1","direction":"higher",
 "diagnostics":{"train_loss":0.128,"validation_loss":0.528,"gap":0.400,
   "train_metrics":{"accuracy":0.973,"macro_f1":0.956},
   "validation_metrics":{"accuracy":0.887,"macro_f1":0.720},
   "n_groups":1,"overfit_flag":false,"drift_flag":false},
 "previous":[{"action":"train","params":{"lr_scale":0.3,"epochs":1},"before":0.714,"after":0.7257,"accepted":true}],
 "rejections":0}
```

### 场景 A-2 连续被拒（关键：必须换方向）

```json
{"dataset":"A","task":"classification","metric":"macro_f1","direction":"higher",
 "diagnostics":{"train_loss":0.128,"validation_loss":0.528,"gap":0.400,
   "validation_metrics":{"macro_f1":0.7198},"n_groups":1,"overfit_flag":true,"drift_flag":false},
 "previous":[
   {"action":"train","params":{"lr_scale":0.3,"epochs":1},"before":0.7198,"after":0.7198,"accepted":false},
   {"action":"train","params":{"lr_scale":0.3,"epochs":2},"before":0.7198,"after":0.7166,"accepted":false}],
 "rejections":2}
```

期望：`lr_scale` 不再是 0.3，或换成 dropout/weight_decay，不能原样重复。

### 场景 B-1 漂移（回归，应偏保守+正则）

```json
{"dataset":"B","task":"regression","metric":"rmse","direction":"lower",
 "diagnostics":{"train_loss":0.089,"validation_loss":0.694,"gap":0.605,
   "train_metrics":{"rmse":230.1},"validation_metrics":{"rmse":651.6},
   "group_metrics":[{"group":["Bearing1_4"],"primary":274.4},{"group":["Bearing2_4"],"primary":358.7},{"group":["Bearing3_4"],"primary":678.9}],
   "n_groups":3,"overfit_flag":true,"drift_flag":true},
 "previous":[],"rejections":0}
```

期望：小步 + 正则（dropout 或 weight_decay），或明确说明为何 no_op。

---

## 五、合格判定（每条都过才算调好）

| 检查 | 合格线 |
|---|---|
| 输出是纯 JSON | 能从第一个 `{` 解析到最后一个 `}`，无多余文字 |
| 字段齐全 | `action`、`hypothesis`、`evidence_ids`、`params` |
| 参数在界内 | epochs 1–4、lr_scale 0.5–1.5，越界即不合格 |
| A-2 会换方向 | 参数与上一轮不同 |
| 同输入三次一致 | 温度 0 + seed 42 下必须完全相同 |
| evidence_ids 不虚构 | 只出现输入里有的字段名 |

## 六、常见坏输出怎么处理

| 现象 | 原因 | 处理 |
|---|---|---|
| 先输出大段推理再给 JSON | thinking 没关 | 关 Enable thinking，或把 token 限制提到 1536 |
| 输出被 ```json 包住 | 模型习惯 | 解析器已能去围栏，但提示词里再强调一次“不要代码块” |
| `params` 缺字段 | 提示词没给模板 | 保留 schema 示例行 |
| 每次参数都不同 | 温度/seed 没生效 | 温度 0、seed 42，重发验证 |
| 直接复述观测不决策 | token 限制太小被截断 | 提到 1024–1536 |

## 七、调好之后怎么搬回代码

控制台的结论 = 代码的默认值，两边必须一致：

| 控制台 | 代码 |
|---|---|
| Temperature 0 | `generate(..., temperature=0.0)` |
| Top p 1.0 | `generate(..., top_p=1.0)` |
| Seed 42 | `generate(..., seed=42)` |
| Enable thinking 关 | `generate(..., thinking=False)` |
| Token 1024 | `qwen_decide(..., max_tokens=1536)`（留余量） |
| System 提示词 | `src/strategy/deciders.py:_build_prompt` |
| 参数边界 | `src/policy/bounds.py:clamp_params` |

改代码前先在控制台把这四类场景跑通，避免代码改完还要来回试。
