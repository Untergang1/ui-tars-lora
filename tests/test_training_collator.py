"""Unit tests for batching the grounding SFT collator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_grounding import GroundingCollator  # noqa: E402


class FakeProcessor:
    def apply_chat_template(self, messages: list[dict[str, object]], **_: object) -> str:
        return "prompt" if len(messages) == 1 else "prompt answer"

    def __call__(self, *, text: list[str], **_: object) -> dict[str, torch.Tensor]:
        is_full = text[0].endswith("answer")
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]] if is_full else [[1, 2, 0], [5, 6, 7]])
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).long()}


class GroundingCollatorTests(unittest.TestCase):
    def test_two_examples_are_batched_and_masked_independently(self) -> None:
        examples = [
            {"image": "one.png", "prompt": "first", "response": "(1, 1)"},
            {"image": "two.png", "prompt": "second", "response": "(2, 2)"},
        ]
        with patch("train_grounding.process_vision_info", return_value=([object(), object()], None)):
            batch = GroundingCollator(FakeProcessor())(examples)

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertTrue(torch.equal(batch["labels"], torch.tensor([[-100, -100, 3, 4], [-100, -100, -100, 8]])))

    def test_empty_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot collate an empty batch"):
            GroundingCollator(FakeProcessor())([])


if __name__ == "__main__":
    unittest.main()
