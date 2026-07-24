#!/usr/bin/env python3
"""Run or score one application's held-out UI grounding evaluation."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import signal
import socket
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from grounding_metrics import aggregate, score_label
from training_config import TrainingConfig, load_training_config, validate_dataset_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA_ENV = Path("/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora")
DEFAULT_PORT = 18001
DEFAULT_GPU = 1
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOKENS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/apps/avantage.yaml")
    parser.add_argument("--responses", type=Path, help="Existing JSONL objects with id and response fields")
    parser.add_argument("--report", type=Path, help="Overrides the timestamped report path")
    parser.add_argument(
        "--adapter",
        type=Path,
        help="LoRA adapter path, resolved from the current working directory; overrides the configured run's adapters/best",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Temporary vLLM port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--gpu", type=int, default=DEFAULT_GPU,
        help=f"GPU index for the temporary vLLM service (default: {DEFAULT_GPU})",
    )
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_responses(path: Path) -> dict[str, object]:
    responses: dict[str, object] = {}
    for row in read_jsonl(path):
        item_id = str(row.get("id", ""))
        if not item_id or "response" not in row:
            raise ValueError("each response must have non-empty id and response fields")
        if item_id in responses:
            raise ValueError(f"duplicate response id: {item_id}")
        responses[item_id] = row["response"]
    return responses


def grouped_aggregate(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {value: aggregate(group) for value, group in sorted(groups.items())}


def default_report_path(output: Path, evaluated_at: datetime) -> Path:
    return output / f"eval_{evaluated_at.strftime('%m%d_%H%M%S')}.json"


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def validate_automatic_args(args: argparse.Namespace) -> None:
    if args.gpu < 0:
        raise ValueError("gpu must be a non-negative integer")
    if args.port == 18000:
        raise ValueError("port 18000 is reserved for the production service")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError("startup and request timeouts must be positive")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if is_port_in_use(args.port):
        raise ValueError(f"port {args.port} is already in use")


def validate_adapter(adapter: Path, app_id: str) -> Path:
    resolved = adapter.resolve()
    if not (resolved / "adapter_config.json").is_file():
        raise ValueError(f"not a PEFT adapter directory: {resolved}")
    metadata_path = resolved / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing application metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid application metadata: {metadata_path}: {exc}") from exc
    if metadata.get("app_id") != app_id:
        raise ValueError(f"adapter belongs to {metadata.get('app_id')!r}, not {app_id!r}")
    return resolved


def resolve_adapter(config: TrainingConfig, requested_adapter: Path | None) -> tuple[Path | None, str]:
    """Choose an explicit adapter or the current run's published best adapter."""
    if requested_adapter is not None:
        return validate_adapter(requested_adapter, config.app_id), "explicit_argument"
    if not config.adapters.exists():
        return None, "adapters_directory_missing"
    if not config.adapters.is_dir():
        raise ValueError(f"adapters path is not a directory: {config.adapters}")
    return validate_adapter(config.adapters / "best", config.app_id), "config_run_name"


def vllm_command(
    config: TrainingConfig, port: int, adapter: Path | None, adapter_source: str
) -> tuple[list[str], str, dict[str, object]]:
    executable = CONDA_ENV / "bin" / "vllm"
    if not executable.is_file():
        raise ValueError(f"missing isolated environment: {executable}; run scripts/create_environment.sh first")
    if not config.model.is_dir():
        raise ValueError(f"base model does not exist: {config.model}; run scripts/copy_model.sh first")
    served_model = f"ui-tars-1.5-{config.app_id}-evaluation"
    command = [
        str(executable),
        "serve",
        str(config.model),
        "--served-model-name",
        served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
    ]
    inference: dict[str, object] = {
        "mode": "native",
        "model": str(config.model),
        "served_model": served_model,
        "adapter": None,
        "adapter_source": adapter_source,
    }
    request_model = served_model
    if adapter is not None:
        lora_name = f"{config.app_id}-grounding"
        command.extend(["--enable-lora", "--lora-modules", f"{lora_name}={adapter}"])
        inference.update({"mode": "lora", "adapter": str(adapter), "request_model": lora_name})
        request_model = lora_name
    return command, request_model, inference


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return payload


def wait_for_service(process: subprocess.Popen[object], base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "service did not accept connections"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"temporary vLLM service exited with status {process.returncode}")
        try:
            get_json(f"{base_url}/v1/models", timeout=2.0)
            return
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"temporary vLLM service did not become ready within {timeout:g}s: {last_error}")


class TemporaryVllmService:
    """Own one evaluation-only vLLM process and its GPU process group."""

    def __init__(self, command: list[str], port: int, startup_timeout: float, gpu: int) -> None:
        self.command = command
        self.port = port
        self.startup_timeout = startup_timeout
        self.gpu = gpu
        self.process: subprocess.Popen[object] | None = None

    def __enter__(self) -> str:
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(self.gpu),
                "HF_HOME": str(PROJECT_ROOT / ".cache/huggingface"),
                "HF_HUB_CACHE": str(PROJECT_ROOT / ".cache/huggingface/hub"),
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        self.process = subprocess.Popen(self.command, env=environment, start_new_session=True)
        base_url = f"http://127.0.0.1:{self.port}"
        try:
            wait_for_service(self.process, base_url, self.startup_timeout)
        except BaseException:
            self.stop()
            raise
        return base_url

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
            self.process.wait()


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def request_response(base_url: str, model: str, label: dict[str, object], timeout: float, max_tokens: int) -> str:
    image_path = Path(str(label["image"]))
    if not image_path.is_file():
        raise ValueError(f"validation image does not exist: {image_path}")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    {"type": "text", "text": str(label["prompt"])},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed for {label['id']}: HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vLLM request failed for {label['id']}: {exc}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"vLLM response for {label['id']} did not contain a chat completion") from exc
    if not isinstance(content, str):
        raise RuntimeError(f"vLLM response for {label['id']} had a non-text completion")
    return content


def infer_responses(config: TrainingConfig, labels: list[dict[str, object]], args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    validate_automatic_args(args)
    adapter, adapter_source = resolve_adapter(config, args.adapter)
    command, request_model, inference = vllm_command(config, args.port, adapter, adapter_source)
    responses: dict[str, object] = {}
    with TemporaryVllmService(command, args.port, args.startup_timeout, args.gpu) as base_url:
        for index, label in enumerate(labels, start=1):
            item_id = str(label["id"])
            print(f"Evaluating {index}/{len(labels)}: {item_id}", flush=True)
            responses[item_id] = request_response(base_url, request_model, label, args.request_timeout, args.max_tokens)
    inference.update(
        {
            "gpu": args.gpu,
            "port": args.port,
            "max_tokens": args.max_tokens,
            "request_timeout_seconds": args.request_timeout,
        }
    )
    return responses, inference


def main() -> None:
    args = parse_args()
    if args.responses is not None and args.adapter is not None:
        raise SystemExit("error: --adapter can only be used when the script performs automatic inference")
    try:
        config = load_training_config(args.config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if not config.validation.is_file() or not config.manifest.is_file():
        raise SystemExit(f"error: processed validation data is missing for {config.app_id}; run prepare_grounding_data.py first")
    manifest = json.loads(config.manifest.read_text(encoding="utf-8"))
    try:
        validate_dataset_manifest(config, manifest)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    labels = read_jsonl(config.validation)
    if any(
        str(label.get("app_id", "")) != config.app_id
        or str(label.get("dataset_version", "")) != config.dataset_version
        for label in labels
    ):
        raise SystemExit("error: validation labels do not all belong to the configured application and dataset version")
    evaluated_at = datetime.now(timezone.utc)
    report_path = args.report or default_report_path(config.output, evaluated_at)
    if args.report is None and report_path.exists():
        raise SystemExit(f"error: timestamped report already exists: {report_path}")
    try:
        if args.responses is not None:
            responses = read_responses(args.responses)
            inference: dict[str, object] = {"mode": "offline_responses", "responses": str(args.responses.resolve())}
        else:
            responses, inference = infer_responses(config, labels, args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    results = [score_label(label, responses.get(str(label["id"]))) for label in labels]
    label_ids = {str(label["id"]) for label in labels}
    report = {
        "app_id": config.app_id,
        "dataset_version": config.dataset_version,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "dataset_manifest": str(config.manifest),
        "bbox_convention": "left/top inclusive; right/bottom exclusive",
        "inference": inference,
        "metrics": aggregate(results),
        "by_app_version": grouped_aggregate(results, "app_version"),
        "unexpected_response_ids": sorted(set(responses) - label_ids),
        "examples": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "app_id": config.app_id, **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
