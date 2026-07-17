"""Tests for bbox hit-rate scoring and failure handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_grounding import aggregate, score_label  # noqa: E402


LABEL = {
    "id": "target",
    "app_id": "avantage",
    "target_coordinate": {"x": 0, "y": 0, "width": 1920, "height": 1080},
    "bbox": {"left": 10, "top": 20, "right": 30, "bottom": 40},
    "bbox_center": {"x": 19.5, "y": 29.5},
    "original_image": {"width": 1920, "height": 1080},
    "app_version": "1.0",
}


class EvaluationTests(unittest.TestCase):
    def test_bbox_boundary_is_left_top_inclusive_and_right_bottom_exclusive(self) -> None:
        hit = score_label(LABEL, "(10, 20)")
        right_edge = score_label(LABEL, "(30, 20)")
        bottom_edge = score_label(LABEL, "(10, 40)")

        self.assertTrue(hit["bbox_hit"])
        self.assertFalse(right_edge["bbox_hit"])
        self.assertFalse(bottom_edge["bbox_hit"])

    def test_unparseable_and_out_of_range_responses_reduce_total_accuracy(self) -> None:
        rows = [score_label(LABEL, "(10, 20)"), score_label(LABEL, "none"), score_label(LABEL, "(1921, 20)")]
        metrics = aggregate(rows)

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["bbox_hits"], 1)
        self.assertEqual(metrics["parseable"], 2)
        self.assertEqual(metrics["in_contract_range"], 1)
        self.assertAlmostEqual(metrics["bbox_accuracy"], 1 / 3)

if __name__ == "__main__":
    unittest.main()
