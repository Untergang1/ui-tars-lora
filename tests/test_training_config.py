"""Regression tests for the YAML-only training configuration interface."""

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
        self.config_path = PROJECT_ROOT / "configs/lora_qlora.yaml"

    def test_default_config_resolves_all_runtime_paths(self) -> None:
        config = load_training_config(self.config_path)

        self.assertEqual(config.model, PROJECT_ROOT / "models/UI-TARS-1.5-7B")
        self.assertEqual(config.train, PROJECT_ROOT / "data/processed/train.jsonl")
        self.assertEqual(config.validation, PROJECT_ROOT / "data/processed/validation.jsonl")
        self.assertTrue(config.language_gradient_checkpointing)
        self.assertFalse(config.vision_gradient_checkpointing)
        self.assertEqual(config.target_modules[0], "q_proj")

    def test_unknown_config_key_is_rejected(self) -> None:
        values = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        values["ignored_setting"] = True
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "unsupported keys: ignored_setting"):
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
