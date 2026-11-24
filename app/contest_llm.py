"""决策层：只走平台网关 rnd.model，不直连任何外部服务，不拼官方提示词。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

DEFAULTS = {"action": "train", "epochs": 3, "learning_rate_scale": 1.0,
            "batch_size_scale": 1.0, "hypothesis": "沿用基线配置做一轮短程继续训练。",
            "change_summary": "epochs 与学习率取基线默认值，不做额外改动。",
            "reason": "模型输出不可解析，回退到保守默认参数。", "source": "fallback"}

_EPOCH_RANGE = (1, 4)
_SCALE_RANGE = (0.5, 1.5)


def _prompt(context: Dict[str, Any]) -> str:
    rejections = context.get("rejections")
    previous = context.get("previous") or {}
    history_note = ""
    if isinstance(rejections, int) and rejections > 0:
        history_note = (
            "\n注意：此前已连续 %d 轮候选未超过父检查点。若上一次的超参数方向无效，本轮必须换方向"
            "（换学习率档位、换 epoch 数、或换正则），不得与上一轮给出相同的组合。\n" % rejections
        )
    if previous.get("epochs") is not None:
        history_note += "上一轮实际使用：epochs=%s，learning_rate_scale=%s，主指标 %s。\n" % (
            previous.get("epochs"), previous.get("learning_rate_scale"), previous.get("metric_value"))
    return (
        "你是工业模型漂移自演化系统的决策器。本轮只做一次判断，不要解释过程。\n"
        "动作固定为 train：始终在平台提供的父检查点基础上继续训练，不新建模型、不使用外部数据。\n"
        "硬性约束：epochs 取 %d..%d；learning_rate_scale 取 %.1f..%.1f（相对基线学习率）；不得改变网络结构、不得引入外部数据。\n"
        "只输出一个 JSON 对象，键为 action、epochs、learning_rate_scale、hypothesis、change_summary、reason。\n"
        "hypothesis 用一句话说明你预计本轮会带来什么变化；change_summary 用一句话说明改了什么。\n"
        + history_note +
        "\n本轮上下文：\n%s"
    ) % (_EPOCH_RANGE[0], _EPOCH_RANGE[1], _SCALE_RANGE[0], _SCALE_RANGE[1],
         json.dumps(context, ensure_ascii=False, indent=1, sort_keys=True))


def _parse(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    body = re.sub(r"^```[a-zA-Z]*|```$", "", text.strip(), flags=re.MULTILINE)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _float(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(low, min(high, number))


def _text(value: Any, fallback: str, limit: int = 1200) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return value.strip()[:limit]


def _diversify(plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """连续被拒时强制换方向，避免逐轮重复同一组超参数。纯确定性规则，不依赖模型输出。"""
    try:
        rejections = int(context.get("rejections") or 0)
    except (TypeError, ValueError):
        rejections = 0
    previous = context.get("previous") or {}
    if rejections < 2 or not isinstance(previous, dict):
        return plan
    try:
        prev_lr = float(previous.get("learning_rate_scale"))
        prev_epochs = int(previous.get("epochs"))
    except (TypeError, ValueError):
        return plan
    same_lr = abs(plan["learning_rate_scale"] - prev_lr) < 1e-9
    same_epochs = plan["epochs"] == prev_epochs
    if not (same_lr and same_epochs):
        return plan  # 已经换了方向，不需要强制干预
    plan["learning_rate_scale"] = _SCALE_RANGE[1] if prev_lr < 1.25 else _SCALE_RANGE[0]
    plan["epochs"] = _EPOCH_RANGE[1] if prev_epochs < 3 else max(_EPOCH_RANGE[0], 2)
    plan["reason"] = _text(
        "%s（连续 %d 轮未超过父检查点，按退火规则强制改变学习率与 epoch 数）" % (plan.get("reason", ""), rejections),
        DEFAULTS["reason"])
    plan["change_summary"] = _text(
        "%s；本轮按退火规则把 epochs 调为 %d、learning_rate_scale 调为 %.2f。"
        % (plan.get("change_summary", ""), plan["epochs"], plan["learning_rate_scale"]),
        DEFAULTS["change_summary"])
    plan["source"] = str(plan.get("source") or "model") + "+anneal"
    return plan


def decide(rnd, context: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """返回 (决策, model_call_id)。调用失败也要给出可执行决策，保证本轮不空转。"""
    messages = [{"role": "user", "content": _prompt(context)}]
    receipt = rnd.model(messages, max_tokens=min(512, rnd.max_output_tokens), temperature=0)
    parsed = _parse(receipt.text)
    call_id = receipt.call_id
    if not parsed:
        plan = dict(DEFAULTS)
        plan["raw_reply"] = receipt.text[:600]
        plan["action"] = "train"
        return _diversify(plan, context), call_id
    action = "no_op" if str(parsed.get("action", "")).strip().lower() in ("no_op", "noop", "skip") else "train"
    plan = {
        "action": action,
        "epochs": _int(parsed.get("epochs"), _EPOCH_RANGE[0], _EPOCH_RANGE[1], DEFAULTS["epochs"]),
        "learning_rate_scale": _float(parsed.get("learning_rate_scale"), _SCALE_RANGE[0], _SCALE_RANGE[1], 1.0),
        "batch_size_scale": 1.0,
        "hypothesis": _text(parsed.get("hypothesis"), DEFAULTS["hypothesis"]),
        "change_summary": _text(parsed.get("change_summary"), DEFAULTS["change_summary"]),
        "reason": _text(parsed.get("reason"), DEFAULTS["reason"]),
        "source": "model",
    }
    return _diversify(plan, context), call_id
