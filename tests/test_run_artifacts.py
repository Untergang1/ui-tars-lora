"""Tests for training artifact isolation without loading model dependencies."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_artifacts import RunArtifacts  # noqa: E402


class FakeModel:
    def __init__(self, marker: str, fail: bool = False) -> None:
        self.marker = marker
        self.fail = fail

    def save_pretrained(self, destination: Path) -> None:
        if self.fail:
            raise RuntimeError("save failed")
        destination.mkdir()
        (destination / "adapter_config.json").write_text("{}", encoding="utf-8")
        (destination / "adapter_model.safetensors").write_text(self.marker, encoding="utf-8")


class RunArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary_directory.name) / "outputs" / "avantage" / "v2"
        self.config = {"app_id": "avantage", "run_name": "v2", "learning_rate": 0.0001}
        self.manifest = {"app_id": "avantage", "input_hash": "abc"}

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def artifacts(self) -> RunArtifacts:
        return RunArtifacts(self.run_dir, self.config, self.manifest, "avantage", "v2")

    def test_new_run_writes_records_and_refuses_a_second_fresh_start(self) -> None:
        artifacts = self.artifacts()
        self.assertIsNone(artifacts.prepare(resume=False))

        self.assertTrue((self.run_dir / "adapters").is_dir())
        self.assertTrue((self.run_dir / "checkpoints").is_dir())
        self.assertEqual(json.loads((self.run_dir / "records" / "run_config.json").read_text(encoding="utf-8")), self.config)
        self.assertEqual(json.loads((self.run_dir / "records" / "status.json").read_text(encoding="utf-8"))["status"], "running")
        with self.assertRaisesRegex(ValueError, "already contains artifacts"):
            self.artifacts().prepare(resume=False)

    def test_new_run_allows_only_the_launcher_log_to_preexist(self) -> None:
        records = self.run_dir / "records"
        records.mkdir(parents=True)
        (records / "train_20260721T120000Z_1.log").write_text("launcher started\n", encoding="utf-8")

        self.assertIsNone(self.artifacts().prepare(resume=False))

    def test_resume_uses_latest_checkpoint_and_requires_frozen_inputs(self) -> None:
        artifacts = self.artifacts()
        artifacts.prepare(resume=False)
        (artifacts.checkpoints / "checkpoint-9").mkdir()
        (artifacts.checkpoints / "checkpoint-12").mkdir()

        resumed = self.artifacts().prepare(resume=True)
        self.assertEqual(resumed, artifacts.checkpoints / "checkpoint-12")
        changed = RunArtifacts(self.run_dir, {**self.config, "learning_rate": 0.001}, self.manifest, "avantage", "v2")
        with self.assertRaisesRegex(ValueError, "frozen run configuration"):
            changed.prepare(resume=True)

    def test_metric_history_selects_best_and_adapter_publish_preserves_existing_copy(self) -> None:
        artifacts = self.artifacts()
        artifacts.prepare(resume=False)
        artifacts.record_metrics(10, 1.0, {"eval_loss": 0.4})
        self.assertTrue(artifacts.should_export_best(0.4))
        self.assertFalse(artifacts.should_export_best(0.5))
        self.assertTrue(artifacts.should_export_best(0.3))

        artifacts.export_adapter(FakeModel("first"), "last", 10, 1.0)
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            artifacts.export_adapter(FakeModel("second", fail=True), "last", 11, 1.1)
        adapter = artifacts.adapters / "last"
        self.assertEqual((adapter / "adapter_model.safetensors").read_text(encoding="utf-8"), "first")
        metadata = json.loads((adapter / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["app_id"], "avantage")
        self.assertEqual(metadata["selection"], "last")
        self.assertEqual(artifacts.best_eval_loss(), 0.4)

    def test_epoch_evaluation_is_a_single_atomic_report_per_epoch(self) -> None:
        artifacts = self.artifacts()
        artifacts.prepare(resume=False)
        summary = {
            "total": 2,
            "bbox_accuracy": 0.5,
            "pixel_distance_to_bbox_center": {"mean": 10.0, "median": 10.0, "p90": 12.0},
        }
        examples = [
            {"id": "one", "bbox_hit": True, "pixel_distance_to_bbox_center": 1.0, "relative_diagonal_error": 0.01},
            {"id": "two", "bbox_hit": False, "pixel_distance_to_bbox_center": None, "relative_diagonal_error": None},
        ]

        path = artifacts.record_epoch_evaluation(1, 10, 0.4, summary, examples)
        artifacts.record_epoch_evaluation(1, 11, 0.3, summary, examples)

        self.assertEqual(path, artifacts.records / "eval" / "epoch_001.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["epoch"], 1)
        self.assertEqual(report["global_step"], 11)
        self.assertEqual(report["summary"]["eval_loss"], 0.3)
        self.assertEqual(report["examples"], examples)


if __name__ == "__main__":
    unittest.main()
