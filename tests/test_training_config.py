"""Regression tests for the per-application training configuration interface."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from training_config import load_training_config, validate_dataset_manifest  # noqa: E402


class TrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = PROJECT_ROOT / "configs/apps/app.template.yaml"

    def test_default_config_resolves_application_paths(self) -> None:
        config = load_training_config(self.config_path)

        self.assertEqual(config.app_id, "example-app")
        self.assertEqual(config.data_root, PROJECT_ROOT / "data/example-app")
        self.assertEqual(config.dataset_version, "v1")
        self.assertEqual(config.dataset_root, PROJECT_ROOT / "data/example-app/v1")
        self.assertEqual(config.train, PROJECT_ROOT / "data/example-app/v1/processed/train.jsonl")
        self.assertEqual(config.validation, PROJECT_ROOT / "data/example-app/v1/processed/validation.jsonl")
        self.assertEqual(config.output, PROJECT_ROOT / "outputs/example-app/baseline")
        self.assertEqual(config.adapters, PROJECT_ROOT / "outputs/example-app/baseline/adapters")
        self.assertEqual(config.checkpoints, PROJECT_ROOT / "outputs/example-app/baseline/checkpoints")
        self.assertEqual(config.records, PROJECT_ROOT / "outputs/example-app/baseline/records")
        self.assertEqual(config.validation_fraction, 0.2)
        self.assertTrue(config.language_gradient_checkpointing)
        self.assertFalse(config.vision_gradient_checkpointing)

    def test_unknown_config_key_is_rejected(self) -> None:
        values = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        values["ignored_setting"] = True
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "unsupported keys: ignored_setting"):
                load_training_config(Path(handle.name))

    def test_application_data_paths_cannot_escape_data_root(self) -> None:
        values = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        values["processed_dir"] = "../other-app/processed"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "processed_dir must stay inside data_root"):
                load_training_config(Path(handle.name))

    def test_data_root_must_match_application_id(self) -> None:
        values = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        values["data_root"] = "data/other-app"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "data_root directory name must match app_id"):
                load_training_config(Path(handle.name))

    def test_dataset_version_must_be_a_positive_v_number(self) -> None:
        values = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        values["dataset_version"] = "release-1"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "dataset_version must use the form"):
                load_training_config(Path(handle.name))

    def test_manifest_must_match_the_selected_dataset_version(self) -> None:
        config = load_training_config(self.config_path)

        with self.assertRaisesRegex(ValueError, "manifest version"):
            validate_dataset_manifest(config, {"app_id": "example-app", "dataset_version": "v2"})

    def test_invalid_checkpointing_value_is_rejected(self) -> None:
        values = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        values["vision_gradient_checkpointing"] = "false"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "vision_gradient_checkpointing must be true or false"):
                load_training_config(Path(handle.name))


if __name__ == "__main__":
    unittest.main()
