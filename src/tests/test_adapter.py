"""适配层单测：批次读取、原子写、no_op 产物。不触发任何训练。

用法：python src/tests/test_adapter.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

import project  # noqa: E402
from adapter.artifact import atomic_write_json, sha256_file, write_no_op  # noqa: E402
from adapter.batch import iter_batches, read_batch  # noqa: E402


def write_batch(path: Path, rows: int, answer: str = "Ethanol") -> Path:
    lines = []
    for index in range(rows):
        lines.append(json.dumps({
            "id": f"s{index}",
            "inputs": {"features": [0.1 * index, 0.2 * index]},
            "answer": answer,
            "meta": {"source_batch": "b1"},
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class AdapterTest(unittest.TestCase):
    def test_batch_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            batch_dir = Path(temp) / "inbox"
            batch_dir.mkdir()
            write_batch(batch_dir / "batch-0001.jsonl", 3)
            write_batch(batch_dir / "batch-0002.jsonl", 2)

            refs = iter_batches(batch_dir)
            self.assertEqual([ref.batch_id for ref in refs], ["batch-0001", "batch-0002"])

            data, info = read_batch(refs[0], "classification", ["Ethanol"])
            self.assertEqual(len(data["ids"]), 3)
            self.assertEqual(info["n_rows"], 3)
            self.assertTrue(info["sha256"].startswith("sha256:"))

    def test_schema_consistent_across_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            batch_dir = Path(temp) / "inbox"
            batch_dir.mkdir()
            write_batch(batch_dir / "batch-0001.jsonl", 2)
            bad = batch_dir / "batch-0002.jsonl"
            bad.write_text(json.dumps({"id": "x", "inputs": {"other": [1.0]}, "answer": "Ethanol"}) + "\n",
                           encoding="utf-8")
            refs = iter_batches(batch_dir)
            first, _ = read_batch(refs[0], "classification", ["Ethanol"])
            with self.assertRaises(ValueError):
                read_batch(refs[1], "classification", ["Ethanol"], first["paths"])

    def test_missing_answer_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            batch_dir = Path(temp) / "inbox"
            batch_dir.mkdir()
            (batch_dir / "batch-0001.jsonl").write_text(
                json.dumps({"id": "x", "inputs": {"features": [1.0]}}) + "\n", encoding="utf-8")
            refs = iter_batches(batch_dir)
            with self.assertRaises(ValueError):
                read_batch(refs[0], "classification", ["Ethanol"])

    def test_atomic_write_and_no_op(self):
        with tempfile.TemporaryDirectory() as temp:
            submit = Path(temp) / "submit"
            target = submit / "round-1" / "A"
            atomic_write_json(target / "x.json", {"ok": True})
            self.assertTrue((target / "x.json").is_file())
            self.assertFalse(list(target.glob("*.tmp")))
            self.assertTrue(sha256_file(target / "x.json").startswith("sha256:"))

            out = write_no_op(submit, "A", 1, current_checkpoint=None,
                              hypothesis="测试", reason="unit")
            payload = json.loads((out / "no_op.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["deploy"])
            self.assertEqual(payload["reason"], "unit")

    def test_relative_paths_only(self):
        path = project.ROOT / "runs" / "current.json"
        self.assertEqual(project.to_store_path(path), "runs/current.json")
        self.assertEqual(project.from_store_path("runs/current.json"), path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
