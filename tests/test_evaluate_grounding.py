"""Tests for bbox hit-rate scoring and Transformers evaluation behavior."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_grounding import (  # noqa: E402
    aggregate,
    default_report_path,
    infer_responses,
    infer_transformers_responses,
    parse_args,
    resolve_adapter,
    score_label,
    validate_adapter,
    validate_common_inference_args,
)


LABEL = {
    "id": "target",
    "app_id": "avantage",
    "target_coordinate": {"x": 0, "y": 0, "width": 100, "height": 80},
    "bbox": {"left": 10, "top": 20, "right": 30, "bottom": 40},
    "bbox_center": {"x": 19.5, "y": 29.5},
    "original_image": {"width": 100, "height": 80},
    "app_version": "1.0",
}


class EvaluationTests(unittest.TestCase):
    def test_bbox_boundary_is_left_top_inclusive_and_right_bottom_exclusive(self) -> None:
        hit = score_label(LABEL, "(10, 20)")
        right_edge = score_label(LABEL, "(30, 20)")
        bottom_edge = score_label(LABEL, "(10, 40)")

        self.assertTrue(hit["bbox_hit"])
        self.assertEqual(hit["predicted_pixel"], {"x": 10, "y": 20})
        self.assertFalse(right_edge["bbox_hit"])
        self.assertFalse(bottom_edge["bbox_hit"])

    def test_unparseable_and_out_of_range_responses_reduce_total_accuracy(self) -> None:
        rows = [score_label(LABEL, "(10, 20)"), score_label(LABEL, "none"), score_label(LABEL, "(100, 20)")]
        metrics = aggregate(rows)

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["bbox_hits"], 1)
        self.assertEqual(metrics["parseable"], 2)
        self.assertEqual(metrics["in_contract_range"], 1)
        self.assertAlmostEqual(metrics["bbox_accuracy"], 1 / 3)

    def test_aggregate_includes_range_rate_and_distance_percentiles(self) -> None:
        rows = [
            {"parseable": True, "in_contract_range": True, "bbox_hit": True, "pixel_distance_to_bbox_center": 0.0, "relative_diagonal_error": 0.0},
            {"parseable": True, "in_contract_range": True, "bbox_hit": False, "pixel_distance_to_bbox_center": 10.0, "relative_diagonal_error": 0.1},
            {"parseable": True, "in_contract_range": True, "bbox_hit": False, "pixel_distance_to_bbox_center": 20.0, "relative_diagonal_error": 0.2},
            {"parseable": False, "in_contract_range": False, "bbox_hit": False},
        ]

        metrics = aggregate(rows)

        self.assertEqual(metrics["in_contract_range_rate"], 0.75)
        self.assertEqual(metrics["pixel_distance_to_bbox_center"], {"mean": 10.0, "median": 10.0, "p90": 18.0})
        relative_errors = metrics["relative_diagonal_error"]
        self.assertAlmostEqual(relative_errors["mean"], 0.1)
        self.assertAlmostEqual(relative_errors["median"], 0.1)
        self.assertAlmostEqual(relative_errors["p90"], 0.18)

    def test_default_report_name_uses_utc_timestamp_format(self) -> None:
        evaluated_at = datetime(2026, 7, 20, 13, 4, 5, tzinfo=timezone.utc)

        report = default_report_path(Path("outputs/avantage/baseline"), evaluated_at)

        self.assertEqual(report, Path("outputs/avantage/baseline/eval_0720_130405.json"))

    def test_transformers_inference_rejects_negative_gpu(self) -> None:
        args = SimpleNamespace(gpu=-1, max_tokens=1)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_common_inference_args(args)

    def test_vllm_options_are_not_accepted(self) -> None:
        for option in ("--backend", "--port", "--startup-timeout", "--request-timeout"):
            with self.subTest(option=option), patch.object(sys, "argv", ["evaluate_grounding.py", option, "1"]):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()

            self.assertEqual(error.exception.code, 2)

    def test_adapter_must_belong_to_the_selected_application(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter = Path(temporary_directory)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "run_metadata.json").write_text('{"app_id": "omnic"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "belongs"):
                validate_adapter(adapter, "avantage")

    def test_relative_adapter_path_is_resolved_to_an_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter = Path(temporary_directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "run_metadata.json").write_text('{"app_id": "avantage"}', encoding="utf-8")
            previous_directory = Path.cwd()
            os.chdir(temporary_directory)
            try:
                resolved = validate_adapter(Path("adapter"), "avantage")
            finally:
                os.chdir(previous_directory)

            self.assertEqual(resolved, adapter.resolve())

    def test_configured_run_uses_its_best_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapters = Path(temporary_directory) / "adapters"
            adapter = adapters / "best"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "run_metadata.json").write_text('{"app_id": "avantage"}', encoding="utf-8")

            resolved, source = resolve_adapter(SimpleNamespace(app_id="avantage", adapters=adapters), None)

            self.assertEqual(resolved, adapter.resolve())
            self.assertEqual(source, "config_run_name")

    def test_missing_adapters_directory_uses_the_native_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapters = Path(temporary_directory) / "adapters"

            adapter, source = resolve_adapter(SimpleNamespace(app_id="avantage", adapters=adapters), None)

            self.assertIsNone(adapter)
            self.assertEqual(source, "adapters_directory_missing")

    def test_existing_adapters_directory_requires_a_valid_best_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapters = Path(temporary_directory) / "adapters"
            adapters.mkdir()

            with self.assertRaisesRegex(ValueError, "not a PEFT adapter directory"):
                resolve_adapter(SimpleNamespace(app_id="avantage", adapters=adapters), None)

    def test_explicit_adapter_overrides_the_configured_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            explicit = root / "explicit"
            explicit.mkdir()
            (explicit / "adapter_config.json").write_text("{}", encoding="utf-8")
            (explicit / "run_metadata.json").write_text('{"app_id": "avantage"}', encoding="utf-8")

            resolved, source = resolve_adapter(
                SimpleNamespace(app_id="avantage", adapters=root / "adapters"), explicit
            )

            self.assertEqual(resolved, explicit.resolve())
            self.assertEqual(source, "explicit_argument")

    @patch("evaluate_grounding.transformers_response", return_value="(10, 20)")
    @patch("evaluate_grounding.load_transformers_model")
    @patch("evaluate_grounding.prepare_transformers_environment")
    def test_transformers_inference_records_exact_adapter_metadata(
        self,
        mocked_environment: MagicMock,
        mocked_loader: MagicMock,
        mocked_response: MagicMock,
    ) -> None:
        config = SimpleNamespace(model=Path("models/UI-TARS-1.5-7B"))
        adapter = Path("outputs/avantage/v4_2/adapters/best")
        args = SimpleNamespace(gpu=1, max_tokens=32)
        mocked_loader.return_value = (MagicMock(), MagicMock(), MagicMock())
        label = {"id": "target", "image": "data/avantage/images/screen.png", "prompt": "Locate target"}

        responses, inference = infer_transformers_responses(config, [label], args, adapter, "config_run_name")

        mocked_environment.assert_called_once_with(1)
        mocked_loader.assert_called_once_with(config, adapter)
        mocked_response.assert_called_once_with(
            mocked_loader.return_value[0],
            mocked_loader.return_value[1],
            mocked_loader.return_value[2],
            label,
            32,
        )
        self.assertEqual(responses, {"target": "(10, 20)"})
        self.assertEqual(inference["backend"], "transformers")
        self.assertEqual(inference["mode"], "transformers_lora")
        self.assertEqual(inference["adapter"], str(adapter))
        self.assertEqual(inference["adapter_source"], "config_run_name")
        self.assertEqual(inference["gpu"], 1)
        self.assertEqual(inference["max_tokens"], 32)

    def test_automatic_inference_always_uses_transformers(self) -> None:
        config = SimpleNamespace(app_id="avantage", adapters=Path("outputs/avantage/v4_2/adapters"))
        args = SimpleNamespace(adapter=None, gpu=1, max_tokens=32)
        adapter = Path("outputs/avantage/v4_2/adapters/best")

        with (
            patch("evaluate_grounding.resolve_adapter", return_value=(adapter, "config_run_name")),
            patch("evaluate_grounding.infer_transformers_responses", return_value=({}, {"backend": "transformers"})) as mocked_infer,
        ):
            _, inference = infer_responses(config, [], args)

        mocked_infer.assert_called_once_with(config, [], args, adapter, "config_run_name")
        self.assertEqual(inference["backend"], "transformers")

    @patch("evaluate_grounding.transformers_response", return_value="(10, 20)")
    @patch("evaluate_grounding.load_transformers_model")
    @patch("evaluate_grounding.prepare_transformers_environment")
    def test_native_transformers_inference_records_native_metadata(
        self,
        mocked_environment: MagicMock,
        mocked_loader: MagicMock,
        mocked_response: MagicMock,
    ) -> None:
        config = SimpleNamespace(model=Path("models/UI-TARS-1.5-7B"))
        args = SimpleNamespace(gpu=1, max_tokens=32)
        mocked_loader.return_value = (MagicMock(), MagicMock(), MagicMock())
        label = {"id": "target", "image": "data/avantage/images/screen.png", "prompt": "Locate target"}

        _, inference = infer_transformers_responses(config, [label], args, None, "adapters_directory_missing")

        mocked_environment.assert_called_once_with(1)
        mocked_loader.assert_called_once_with(config, None)
        mocked_response.assert_called_once()
        self.assertEqual(inference["mode"], "transformers_native")
        self.assertIsNone(inference["adapter"])
        self.assertEqual(inference["adapter_source"], "adapters_directory_missing")


if __name__ == "__main__":
    unittest.main()
