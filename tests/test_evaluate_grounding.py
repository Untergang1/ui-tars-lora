"""Tests for bbox hit-rate scoring and failure handling."""

from __future__ import annotations

import json
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
    TemporaryVllmService,
    aggregate,
    default_report_path,
    request_response,
    resolve_adapter,
    score_label,
    validate_adapter,
    validate_automatic_args,
    vllm_command,
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

    def test_automatic_mode_rejects_the_production_port(self) -> None:
        args = SimpleNamespace(gpu=1, port=18000, startup_timeout=1, request_timeout=1, max_tokens=1)

        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_automatic_args(args)

    def test_automatic_mode_rejects_negative_gpu(self) -> None:
        args = SimpleNamespace(gpu=-1, port=18001, startup_timeout=1, request_timeout=1, max_tokens=1)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_automatic_args(args)

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

    def test_inference_records_adapter_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = root / "environment"
            executable = environment / "bin" / "vllm"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            config = SimpleNamespace(app_id="avantage", model=model)

            with patch("evaluate_grounding.CONDA_ENV", environment):
                _, request_model, inference = vllm_command(config, 18001, None, "adapters_directory_missing")
                _, lora_request_model, lora_inference = vllm_command(
                    config, 18001, root / "adapter", "config_run_name"
                )

            self.assertEqual(request_model, "ui-tars-1.5-avantage-evaluation")
            self.assertEqual(inference["mode"], "native")
            self.assertIsNone(inference["adapter"])
            self.assertEqual(inference["adapter_source"], "adapters_directory_missing")
            self.assertEqual(lora_request_model, "avantage-grounding")
            self.assertEqual(lora_inference["mode"], "lora")
            self.assertEqual(lora_inference["adapter"], str(root / "adapter"))
            self.assertEqual(lora_inference["adapter_source"], "config_run_name")

    def test_request_response_sends_the_validation_image_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image = Path(temporary_directory) / "screen.png"
            image.write_bytes(b"not-a-real-png")
            response = MagicMock()
            response.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "(10, 20)"}}]}
            ).encode("utf-8")
            response.__enter__.return_value = response
            label = {"id": "target", "image": str(image), "prompt": "Query:target\n"}

            with patch("evaluate_grounding.urlopen", return_value=response) as mocked_urlopen:
                completion = request_response("http://127.0.0.1:18001", "native", label, 10, 32)

            request = mocked_urlopen.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(completion, "(10, 20)")
            self.assertEqual(payload["model"], "native")
            self.assertEqual(payload["messages"][0]["content"][1]["text"], "Query:target\n")
            self.assertTrue(payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))

    @patch("evaluate_grounding.os.killpg")
    @patch("evaluate_grounding.os.getpgid", return_value=123)
    @patch("evaluate_grounding.wait_for_service")
    @patch("evaluate_grounding.subprocess.Popen")
    def test_temporary_service_stops_its_process_group(
        self,
        mocked_popen: MagicMock,
        mocked_wait: MagicMock,
        mocked_getpgid: MagicMock,
        mocked_killpg: MagicMock,
    ) -> None:
        process = MagicMock()
        process.pid = 123
        process.poll.return_value = None
        mocked_popen.return_value = process

        with TemporaryVllmService(["vllm", "serve"], 18001, 5, 2) as base_url:
            self.assertEqual(base_url, "http://127.0.0.1:18001")

        mocked_wait.assert_called_once_with(process, "http://127.0.0.1:18001", 5)
        self.assertEqual(mocked_popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
        mocked_getpgid.assert_called_once_with(123)
        mocked_killpg.assert_called_once()
        process.wait.assert_called_once_with(timeout=30)

    @patch("evaluate_grounding.os.killpg")
    @patch("evaluate_grounding.os.getpgid", return_value=123)
    @patch("evaluate_grounding.wait_for_service", side_effect=RuntimeError("not ready"))
    @patch("evaluate_grounding.subprocess.Popen")
    def test_temporary_service_stops_when_startup_fails(
        self,
        mocked_popen: MagicMock,
        mocked_wait: MagicMock,
        mocked_getpgid: MagicMock,
        mocked_killpg: MagicMock,
    ) -> None:
        process = MagicMock()
        process.pid = 123
        process.poll.return_value = None
        mocked_popen.return_value = process

        with self.assertRaisesRegex(RuntimeError, "not ready"):
            with TemporaryVllmService(["vllm", "serve"], 18001, 5, 1):
                pass

        mocked_wait.assert_called_once()
        mocked_getpgid.assert_called_once_with(123)
        mocked_killpg.assert_called_once()
        process.wait.assert_called_once_with(timeout=30)

if __name__ == "__main__":
    unittest.main()
