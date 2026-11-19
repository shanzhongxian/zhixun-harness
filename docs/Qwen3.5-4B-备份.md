# Qwen3.5-4B 版本保留（备份）

训练师模型从 `Qwen/Qwen3.5-4B` 改为 `Qwen/Qwen3.8-27B` 之前的原始记录。
本文件保留 4B 版本的原始引用位置与完整代码，便于回退或对照。

## 原始模型 ID

`Qwen/Qwen3.5-4B`

## 改动涉及的 4 处（改动前原始内容）

1. `训练.py` 第 7 行：

   ```python
   sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3.5-4B")
   ```

2. `harness.py` 第 188 行：

   ```python
   parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
   ```

3. `strategy/deciders.py` 第 88 行：

   ```python
   def qwen_decide(diagnostics, profile, *, model_id="Qwen/Qwen3.5-4B", max_tokens=512):
   ```

4. `selftest.py` 第 69 行：

   ```python
   min_improvement=1e-6, model_id="Qwen/Qwen3.5-4B")
   ```

## 训练.py 原始完整内容（PyTRiO 4B 示例）

```python
import pytrio as trio

# 1. 与TRIO建立连接
service_client = trio.ServiceClient()

# 2. 创建1个推理客户端
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3.5-4B")

# 3. 获取Tokenizer并对输入文本进行预处理
print("Loading tokenizer...")
tokenizer = sampling_client.get_tokenizer()
messages=[{"role": "user", "content": "What's your name?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(input_text)
print("tokenizer finish")

# 4. 推理
params = trio.SamplingParams(max_tokens=50, seed=42, temperature=0.7)
response = sampling_client.sample(
    prompt=trio.ModelInput.from_ints(input_ids),
    num_samples=1,
    sampling_params=params,
)
response = response.result()

print(f"{repr(response.sequences[0].text)}")
```
