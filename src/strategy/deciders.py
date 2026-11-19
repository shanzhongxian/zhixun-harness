"""决策层：规则 / 模拟训练师 / 真实训练师（Qwen via PyTRiO）。

三者统一返回 Decision（见 schemas.py）。真实 Qwen 调用失败时返回 None，
由 harness 降级到规则决策。
"""
import json
import re

from .schemas import Decision, ACTIONS, LR_SCALE_DEFAULTS, PRIMARY_METRIC, DIRECTION


def rule_decide(diagnostics, profile, *, lr_scale=None, epochs=5, patience=2):
    """确定性规则决策器：先诊断，再决定 train / no_op。第 1 版联调基准。"""
    if lr_scale is None:
        lr_scale = LR_SCALE_DEFAULTS[profile]
    if diagnostics.get("batch_drift_flag"):
        return Decision("train",
                        "新批次主指标明显差于公开验证集，疑似漂移，用小步长适配并加 L2",
                        evidence_ids=["batch_drift_flag", "batch_metrics"],
                        params={"lr_scale": min(lr_scale, 0.3), "epochs": min(epochs, 3), "patience": patience, "weight_decay": 1e-4},
                        source="rule")
    if diagnostics.get("overfit_flag"):
        return Decision("no_op",
                        "训练/验证损失差距超过阈值，疑似过拟合，停止续训以避免退化",
                        evidence_ids=["gap", "overfit_flag"], params={}, source="rule")
    if diagnostics.get("drift_flag"):
        return Decision("train",
                        "验证集分组主指标差异大，疑似分布漂移，用小步长适配并加 L2 防过拟合",
                        evidence_ids=["drift_flag", "group_metrics"],
                        params={"lr_scale": min(lr_scale, 0.3), "epochs": min(epochs, 3), "patience": patience, "weight_decay": 1e-4},
                        source="rule")
    return Decision("train",
                    "未观察到明显过拟合或漂移，小步续训尝试提升",
                    evidence_ids=["gap", "drift_flag"],
                    params={"lr_scale": lr_scale, "epochs": epochs, "patience": patience},
                    source="rule")


def mock_decide(diagnostics, profile, *, lr_scale=None, epochs=5, patience=2):
    """模拟训练师：行为像 LLM 但确定性，用于在无真实 Qwen 时联调门禁/回退/降级。"""
    if lr_scale is None:
        lr_scale = LR_SCALE_DEFAULTS[profile]
    if diagnostics.get("overfit_flag"):
        return Decision("no_op",
                        "模拟训练师：验证损失高于训练损失较多，判为过拟合，不再续训",
                        evidence_ids=["gap"], params={}, source="mock")
    return Decision("train",
                    "模拟训练师：主指标仍有优化空间，尝试降低学习率续训",
                    evidence_ids=["gap", "group_metrics"],
                    params={"lr_scale": lr_scale, "epochs": epochs, "patience": patience},
                    source="mock")


def _build_prompt(diagnostics, profile, *, attempts=0, rejections=0) -> str:
    from .schemas import PARAMS_SCHEMA
    tier1 = {k: v for k, v in PARAMS_SCHEMA.items() if v["tier"] == 1}
    primary = diagnostics.get("primary_metric", "")
    current = diagnostics.get("validation_metrics", {}).get(primary)
    history = ""
    if rejections > 0:
        history = (
            f"注意：此前连续 {rejections} 次候选被门禁拒绝。若上一步参数无效，请换方向"
            "（如提高学习率、换损失或加正则），不要重复同样的保守设置。\n"
        )
    return (
        "你是工业模型训练师，负责决定「是否继续训练」以及「用什么参数训练」。\n"
        f"目标：在验证主指标不退化（当前 {primary}={current}）的前提下尽量提升；发现过拟合/漂移时保守。\n"
        + history
        + "观测报告（JSON）：\n" + json.dumps(diagnostics, ensure_ascii=False, default=str) + "\n\n"
        "可调旋钮（只允许这些键与范围）：\n" + json.dumps(tier1, ensure_ascii=False) + "\n\n"
        "请只输出一个 JSON 对象，字段为：\n"
        '{"action": "train|no_op|rollback", "hypothesis": "一句话说明原因", '
        '"evidence_ids": ["引用的观测字段"], "params": {"lr_scale": 0.3, "epochs": 5, "patience": 2}}\n'
        "规则：overfit_flag=true 或训练/验证差距大 → 可提高 dropout/weight_decay，或 no_op；"
        "B 类不稳定 → loss=huber；C 类高估 → loss=asymmetric；否则小步 train。\n"
        "门禁：改动只有在验证主指标（A=macro_f1 越大越好；B/C=rmse 越小越好）严格改善时才会被采纳，"
        "否则被拒、保留当前模型。注意 huber/asymmetric 主要改善稳定性/NASA 分，可能不提升 RMSE，谨慎使用；"
        "若判断当前改动难以提升主指标，优先小幅调学习率/正则，避免乱换损失。\n"
        "只输出 JSON 对象本身，不要输出思考过程、解释或 markdown 代码块。"
    )


def parse_decision_json(text) -> dict:
    """从 LLM 文本里稳健地抽出决策 JSON。

    优先返回含 `action` 键的对象（忽略「思考过程」里引用的观测 JSON 等干扰块）。
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    candidates = []
    for match in re.finditer(r"\{", cleaned):
        start = match.start()
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except Exception:
                        pass
                    break
    for obj in candidates:
        if "action" in obj:
            return obj
    return candidates[0] if candidates else None


def qwen_decide(diagnostics, profile, *, model_id="Qwen/Qwen3.8-27B", max_tokens=1536, attempts=0, rejections=0):
    """真实训练师（PyTRiO）。失败/超时/解析失败时返回 None，由调用方降级到规则。

    参数对着 PyTRiO 控制台调：温度 0、top_p 1.0、关闭 thinking、seed 42。
    关掉 thinking 后 1536 token 足够容纳完整 JSON；若模板不支持关思考，
    27B 仍会先推理，此时靠 1536 的余量和 parse_decision_json 的容错兜住。
    """
    prompt = _build_prompt(diagnostics, profile, attempts=attempts, rejections=rejections)
    try:
        from pytrio_client import generate  # noqa: E402 延迟导入，复用凭据与客户端逻辑
        raw = generate(prompt, model_id=model_id, max_tokens=max_tokens,
                       temperature=0.0, top_p=1.0, thinking=False)["text"]
        parsed = parse_decision_json(raw)
        if parsed is not None:
            parsed["source"] = "qwen"
        return parsed
    except Exception:
        return None
