"""app/ 适配层的纯函数单测（不需要平台、不需要训练）。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import contest_data  # noqa: E402
import contest_llm  # noqa: E402
import contest_submission as cs  # noqa: E402


def test_dataset_key():
    assert contest_data.dataset_key("A") == "A"
    assert contest_data.dataset_key("cpu-b-rul-20261110-v1") == "B"
    assert contest_data.dataset_key("cpu-a-guide-20261110-v1") == "A"
    assert contest_data.dataset_key("CPU-C-GUIDE") == "C"
    assert contest_data.dataset_key("") == "A"
    assert contest_data.dataset_key(None, default="C") == "C"


def test_vector_is_order_stable():
    inputs = {"b": 2, "a": {"y": 1, "x": 0.5}}
    paths, values = contest_data.flatten(inputs)
    assert contest_data.vector(inputs, paths) == values
    assert contest_data.vector({"a": {"x": 1}}, paths) == [1.0, 0.0, 0.0]


def test_parse_variants():
    assert contest_llm._parse('```json\n{"epochs": 2}\n```') == {"epochs": 2}
    assert contest_llm._parse("思考：{not json}") == {}
    assert contest_llm._parse("") == {}
    assert contest_llm._parse("前缀 {\"a\": 1} 后缀") == {"a": 1}


class _Receipt:
    def __init__(self, text, call_id):
        self.text = text
        self.call_id = call_id


class _Round:
    def __init__(self, text):
        self.max_output_tokens = 512
        self._text = text

    def model(self, messages, **options):
        assert messages and messages[0]["role"] == "user"
        assert options["max_tokens"] <= self.max_output_tokens
        return _Receipt(self._text, "call-1")


def test_decide_clamps_and_falls_back():
    plan, call_id = contest_llm.decide(_Round('{"action":"train","epochs":99,"learning_rate_scale":9}'), {})
    assert call_id == "call-1"
    assert plan["epochs"] == 4 and plan["learning_rate_scale"] == 1.5 and plan["source"] == "model"
    plan, _ = contest_llm.decide(_Round("我觉得可以训练"), {})
    assert plan["action"] == "train" and plan["source"] == "fallback" and plan["epochs"] == 3
    plan, _ = contest_llm.decide(_Round('{"action":"no_op"}'), {})
    assert plan["action"] == "no_op"


class _StubModel:
    def __init__(self, task, values):
        self.meta = {"task": task, "classes": ["a", "b"] if task == "classification" else [], "paths": []}
        self._values = values

    def predict(self, matrix):
        return np.asarray(self._values, dtype=np.float64)


class _Sample:
    def __init__(self, task):
        self.task = task
        self.inputs = {"x": 1}


def _with_stub(task, values, call):
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        path = handle.name
    os.environ["CONTEST_ARTIFACT_PATH"] = path
    original = cs._model
    cs._model = lambda checkpoint: _StubModel(task, values)
    try:
        return call()
    finally:
        cs._model = original
        os.unlink(path)


def test_predict_breaks_ties_and_clamps_regression():
    outputs = _with_stub("classification", [[0.5, 0.5]], lambda: cs.predict(_Sample("classification")))
    scores = [item["score"] for item in outputs]
    assert len(outputs) == 2 and scores.count(max(scores)) == 1
    assert all(0.0 <= score <= 1.0 for score in scores)
    value = _with_stub("regression", [-3.0], lambda: cs.predict(_Sample("regression")))
    assert value["value"] == 0.0


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("ok", name)
    print("all app unit checks passed")
