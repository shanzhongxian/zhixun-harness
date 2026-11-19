"""PyTRiO 接入：凭据探测 + 模型可用性检查（默认不做任何训练）。

凭据来源（按优先级）：
1. 环境变量 PYTRIO_API_KEY / TRIO_API_KEY
2. ~/.pytrio/config.toml（由 `pytrio` 登录流程写入）

国内评测环境无外网，凭据只在本机联调/训练时使用，绝不会写进镜像。
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
ALTERNATIVE_MODEL = "Qwen/Qwen3.8-27B"


def credential_status() -> dict:
    env_key = os.environ.get("PYTRIO_API_KEY") or os.environ.get("TRIO_API_KEY") or ""
    config_path = Path.home() / ".pytrio" / "config.toml"
    return {
        "env_key_present": bool(env_key),
        "config_present": config_path.is_file(),
        "config_path": str(config_path),
        "base_url_env": os.environ.get("PYTRIO_BASE_URL", ""),
        "model": os.environ.get("PYTRIO_MODEL", DEFAULT_MODEL),
    }


def build_client(api_key: str | None = None):
    import pytrio

    return pytrio.ServiceClient(api_key=api_key or os.environ.get("PYTRIO_API_KEY") or None)


def probe(model_id: str = DEFAULT_MODEL) -> dict:
    """只验证能不能建起采样客户端，不下载权重、不训练。"""
    status = credential_status()
    if not status["env_key_present"] and not status["config_present"]:
        return {
            "ok": False,
            "stage": "credentials",
            "model": model_id,
            "hint": "缺少 API key：设置 PYTRIO_API_KEY，或先完成 pytrio 登录写入 ~/.pytrio/config.toml",
            "status": status,
        }
    try:
        client = build_client()
        sampling = client.create_sampling_client(base_model=model_id)
        tokenizer = sampling.get_tokenizer()
        return {
            "ok": True,
            "stage": "sampling_client",
            "model": model_id,
            "tokenizer": type(tokenizer).__name__,
            "status": status,
        }
    except Exception as error:  # noqa: BLE001 - 探测要把任何失败都讲清楚
        return {
            "ok": False,
            "stage": "client",
            "model": model_id,
            "error": f"{type(error).__name__}: {error}",
            "hint": "检查 key 是否有效、base_url 是否正确、模型 ID 是否在平台注册表里",
            "status": status,
        }


def generate(prompt: str, model_id: str = DEFAULT_MODEL, max_tokens: int = 512, *,
             temperature: float = 0.0, top_p: float = 1.0, seed: int = 42,
             thinking: bool = False, stop=None, timeout: float = 120.0) -> dict:
    """一次推理（决策用，不是训练）。会产生少量 token 消耗；timeout 秒未返回则抛 TimeoutError。

    默认值对着 PyTRiO 控制台调好的一套：温度 0（决策可复现）、top_p 1.0、
    关闭思考（否则 27B 先输出推理文本再给 JSON，容易截断/解析失败）、seed 42。
    控制台上对应：Temperature 0、Top p 1.0、Enable thinking 关、Seed 42、停止词留空。
    """
    import pytrio

    client = build_client()
    sampling = client.create_sampling_client(base_model=model_id)
    tokenizer = sampling.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    try:  # Qwen3 系模板支持 enable_thinking；老模板不认时退回默认
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=thinking)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    params = pytrio.SamplingParams(max_tokens=max_tokens, seed=seed, temperature=temperature,
                                   top_p=top_p, stop=stop)
    future = sampling.sample(
        prompt=pytrio.ModelInput.from_ints(tokenizer.encode(text)),
        num_samples=1,
        sampling_params=params,
    )
    response = future.result(timeout=timeout)
    text = response.sequences[0].text or ""
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(marker):
            text = text[: -len(marker)]
    return {"model": model_id, "text": text.rstrip(),
            "params": {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                       "seed": seed, "thinking": thinking, "stop": stop}}
