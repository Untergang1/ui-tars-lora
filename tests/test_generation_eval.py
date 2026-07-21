"""Tests for lightweight generation evaluation during training."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_artifacts import RunArtifacts  # noqa: E402
from train_grounding import EpochGenerationEvaluator  # noqa: E402


LABEL = {
    "id": "target",
    "image": "screen.png",
    "prompt": "Query:target\n",
    "target_coordinate": {"x": 10, "y": 20, "width": 100, "height": 80},
    "bbox": {"left": 10, "top": 20, "right": 30, "bottom": 40},
    "bbox_center": {"x": 19.5, "y": 29.5},
    "original_image": {"width": 100, "height": 80},
    "app_version": "1.0",
}


class FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None

    def apply_chat_template(self, messages: list[dict[str, object]], **_: object) -> str:
        self.messages = messages
        return "prompt"

    def __call__(self, **_: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def batch_decode(self, _: torch.Tensor, **__: object) -> list[str]:
        return ["(10, 20)"]


class FakeModel:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.training = True
        self.generate_kwargs: dict[str, object] | None = None

    def parameters(self):
        return iter([self.parameter])

    def eval(self) -> "FakeModel":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "FakeModel":
        self.training = mode
        return self

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


class GenerationEvaluationTests(unittest.TestCase):
    def test_completed_epoch_only_accepts_integer_epochs(self) -> None:
        self.assertEqual(EpochGenerationEvaluator.completed_epoch(2.0), 2)
        self.assertIsNone(EpochGenerationEvaluator.completed_epoch(1.5))
        self.assertIsNone(EpochGenerationEvaluator.completed_epoch(None))

    def test_generation_writes_compact_report_and_restores_training_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "outputs" / "avantage" / "baseline"
            artifacts = RunArtifacts(run_dir, {}, {"app_id": "avantage"}, "avantage", "baseline")
            artifacts.prepare(resume=False)
            processor = FakeProcessor()
            model = FakeModel()
            evaluator = EpochGenerationEvaluator(processor, SimpleNamespace(rows=[LABEL]), artifacts)

            with patch("train_grounding.process_vision_info", return_value=([object()], None)):
                path, summary = evaluator.evaluate(model, epoch=1, global_step=7, eval_loss=0.25)

            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(model.training)
            self.assertEqual(model.generate_kwargs["max_new_tokens"], 32)
            self.assertFalse(model.generate_kwargs["do_sample"])
            self.assertEqual(processor.messages[0]["content"][1]["text"], LABEL["prompt"])
            self.assertNotIn("response", report["examples"][0])
            self.assertEqual(report["summary"]["eval_loss"], 0.25)
            self.assertEqual(summary["bbox_accuracy"], 1.0)
            self.assertEqual(report["examples"][0]["bbox_hit"], True)


if __name__ == "__main__":
    unittest.main()
