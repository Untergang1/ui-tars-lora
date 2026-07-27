#!/usr/bin/env python3
"""Run or score one application's held-out UI grounding evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grounding_metrics import aggregate, score_label
from training_config import TrainingConfig, load_training_config, validate_dataset_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU = 1
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
    parser.add_argument(
        "--gpu", type=int, default=DEFAULT_GPU,
        help=f"GPU index for Transformers inference (default: {DEFAULT_GPU})",
    )
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


def validate_common_inference_args(args: argparse.Namespace) -> None:
    if args.gpu < 0:
        raise ValueError("gpu must be a non-negative integer")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")

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


def prepare_transformers_environment(gpu: int) -> None:
    """Select the evaluation GPU before importing CUDA-aware libraries."""
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        raise RuntimeError("Transformers evaluation must start before CUDA is initialized")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HOME": str(PROJECT_ROOT / ".cache/huggingface"),
            "HF_HUB_CACHE": str(PROJECT_ROOT / ".cache/huggingface/hub"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )


def load_transformers_model(config: TrainingConfig, adapter: Path | None) -> tuple[Any, Any, Any]:
    """Load the local QLoRA base model and optionally attach its PEFT adapter."""
    if not config.model.is_dir():
        raise ValueError(f"base model does not exist: {config.model}; run scripts/copy_model.sh first")
    try:
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Transformers evaluation dependencies are missing; run scripts/create_environment.sh") from exc

    from xformers_vision import enable_xformers_vision_attention

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(config.model, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        config.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    enable_xformers_vision_attention(model)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, processor, process_vision_info


def transformers_response(model: Any, processor: Any, process_vision_info: Any, label: dict[str, object], max_tokens: int) -> str:
    image_path = Path(str(label["image"]))
    if not image_path.is_file():
        raise ValueError(f"validation image does not exist: {image_path}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": str(label["prompt"])},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    generated = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_tokens,
        use_cache=True,
    )
    return processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def infer_transformers_responses(
    config: TrainingConfig,
    labels: list[dict[str, object]],
    args: argparse.Namespace,
    adapter: Path | None,
    adapter_source: str,
) -> tuple[dict[str, object], dict[str, object]]:
    validate_common_inference_args(args)
    prepare_transformers_environment(args.gpu)
    model, processor, process_vision_info = load_transformers_model(config, adapter)
    responses: dict[str, object] = {}
    for index, label in enumerate(labels, start=1):
        item_id = str(label["id"])
        print(f"Evaluating {index}/{len(labels)}: {item_id}", flush=True)
        responses[item_id] = transformers_response(model, processor, process_vision_info, label, args.max_tokens)
    return responses, {
        "mode": "transformers_lora" if adapter is not None else "transformers_native",
        "backend": "transformers",
        "model": str(config.model),
        "adapter": str(adapter) if adapter is not None else None,
        "adapter_source": adapter_source,
        "gpu": args.gpu,
        "max_tokens": args.max_tokens,
    }


def infer_responses(config: TrainingConfig, labels: list[dict[str, object]], args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    adapter, adapter_source = resolve_adapter(config, args.adapter)
    return infer_transformers_responses(config, labels, args, adapter, adapter_source)


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
