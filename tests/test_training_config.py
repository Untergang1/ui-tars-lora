"""Regression tests for the per-application training configuration interface."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from training_config import load_training_config  # noqa: E402


class TrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = PROJECT_ROOT / "configs/apps/avantage.yaml"

    def test_default_config_resolves_application_paths(self) -> None:
        config = load_training_config(self.config_path)

        self.assertEqual(config.app_id, "avantage")
        self.assertEqual(config.data_root, PROJECT_ROOT / "data/software/avantage")
        self.assertEqual(config.train, PROJECT_ROOT / "data/software/avantage/processed/train.jsonl")
        self.assertEqual(config.validation, PROJECT_ROOT / "data/software/avantage/processed/validation.jsonl")
        self.assertEqual(config.output, PROJECT_ROOT / "outputs/avantage/baseline")
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
        values["data_root"] = "data/software/other-app"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "data_root directory name must match app_id"):
                load_training_config(Path(handle.name))

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
