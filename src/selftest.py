"""三条关键路径自测 + 一次端到端跑通（A，规则决策）。

路径：
  1) 非法决策 → 降级 no_op（validate_decision 确定性断言）
  2) 门禁方向比较 → 候选通过 / 退化被拒（合成指标确定性断言）
  3) Qwen 不可用 → 降级规则（monkeypatch，不触发真实下载/调用）
  4) 端到端一轮（A，规则决策）→ 打印实际 result（accepted/rejected/no_op）

用法：python selftest.py
"""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent          # src/
ROOT = SRC.parent                              # 包根目录（智训引擎/）
sys.path.insert(0, str(SRC))

# 用独立目录隔离状态，避免污染正式 runs/。
os.environ["HARNESS_RUNS_DIR"] = str(ROOT / ".selftest" / "runs")

from strategy.schemas import validate_decision, DIRECTION
from strategy.deciders import parse_decision_json


def test_fallback():
    bad = {"action": "explode", "hypothesis": "", "evidence_ids": [1], "params": {"epochs": 999999}}
    clean, errors = validate_decision(bad, "A")
    assert clean.action == "no_op", f"非法 action 应降级 no_op，得到 {clean.action}"
    assert errors, "应产生错误信息"
    assert clean.params.get("epochs") == 200, "epochs 越界应夹到上限 200"
    print("PASS 非法决策降级:", "; ".join(errors))


def test_parse_json():
    raw = "```json\n{\"action\":\"train\",\"hypothesis\":\"x\",\"evidence_ids\":[],\"params\":{\"lr_scale\":0.3}}\n```"
    parsed = parse_decision_json(raw)
    assert parsed and parsed["action"] == "train", parsed
    assert parse_decision_json("no json here") is None
    print("PASS LLM 文本解析 JSON")


def test_gate_direction():
    # A：macro_f1 越大越好
    assert DIRECTION["A"] * (0.72 - 0.71) > 1e-6            # 改善 → 通过
    assert not (DIRECTION["A"] * (0.70 - 0.71) > 1e-6)     # 退化 → 拒绝
    # B/C：rmse 越小越好
    assert DIRECTION["B"] * (50.0 - 52.0) > 1e-6           # rmse 下降 → 改善
    assert not (DIRECTION["B"] * (54.0 - 52.0) > 1e-6)     # rmse 上升 → 退化
    print("PASS 门禁方向比较（A 越大越好，B/C 越小越好）")


def test_qwen_fallback_to_rule():
    import harness
    original = harness.qwen_decide
    harness.qwen_decide = lambda *a, **k: None  # 模拟 Qwen 不可用
    try:
        d = harness._decide("qwen", {"profile": "A", "overfit_flag": False, "drift_flag": False}, "A",
                            lr_scale=0.3, epochs=5, patience=2, model_id="x")
        assert d.source == "rule", d.source
        print("PASS Qwen 不可用 → 降级规则:", d.action, d.source)
    finally:
        harness.qwen_decide = original


def test_e2e():
    import harness
    rec = harness.run_round("A", "rule", lr_scale=0.3, epochs=5, patience=2,
                            batch_size=None, output_dir=".selftest/rounds",
                            min_improvement=1e-6, model_id="Qwen/Qwen3.8-27B")
    p = rec["diagnostics"]["primary_metric"]
    print(f"INFO 端到端 A：action={rec['action']} source={rec['decision_source']} "
          f"result={rec['result']} {p}={rec['diagnostics']['validation_metrics'][p]:.6f}")
    print("INFO 分组主指标:", [(g["group"], round(g["primary"], 4)) for g in rec["diagnostics"]["group_metrics"]])


if __name__ == "__main__":
    test_fallback()
    test_parse_json()
    test_gate_direction()
    test_qwen_fallback_to_rule()
    test_e2e()
    print("\n自测完成。")
