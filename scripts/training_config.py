"""Validated YAML configuration for the UI-TARS grounding training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingConfig:
    """The complete, resolved configuration consumed by ``train_grounding.py``."""

    config_path: Path
    model: Path
    train: Path
    validation: Path
    output: Path
    seed: int
    use_4bit_quantization: bool
    quant_type: str
    compute_dtype: str
    language_gradient_checkpointing: bool
    vision_gradient_checkpointing: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    logging_steps: int
    save_strategy: str
    evaluation_strategy: str
    save_total_limit: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    exclude_modules: str

    def as_json(self) -> dict[str, object]:
        """Return a JSON-safe snapshot with all paths fully resolved."""
        values = asdict(self)
        for key in ("config_path", "model", "train", "validation", "output"):
            values[key] = str(values[key])
        return values


REQUIRED_KEYS = {
    "model_name_or_path",
    "dataset_dir",
    "train_file",
    "validation_file",
    "output_dir",
    "seed",
    "use_4bit_quantization",
    "quant_type",
    "compute_dtype",
    "language_gradient_checkpointing",
    "vision_gradient_checkpointing",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "logging_steps",
    "save_strategy",
    "evaluation_strategy",
    "save_total_limit",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "exclude_modules",
}


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_string(values: dict[str, Any], key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_bool(values: dict[str, Any], key: str) -> bool:
    value = values[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be true or false")
    return value


def _require_positive_int(values: dict[str, Any], key: str) -> int:
    value = values[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _require_positive_float(values: dict[str, Any], key: str) -> float:
    value = values[key]
    if type(value) not in (int, float) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


def load_training_config(path: Path) -> TrainingConfig:
    """Load one complete YAML config and reject values that training cannot honor."""
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"training config does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {config_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError("training config must contain a top-level mapping")

    keys = set(loaded)
    missing = sorted(REQUIRED_KEYS - keys)
    unexpected = sorted(keys - REQUIRED_KEYS)
    if missing:
        raise ValueError(f"training config is missing required keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"training config has unsupported keys: {', '.join(unexpected)}")

    model_name = _require_string(loaded, "model_name_or_path")
    dataset_dir = _require_string(loaded, "dataset_dir")
    train_file = _require_string(loaded, "train_file")
    validation_file = _require_string(loaded, "validation_file")
    output_dir = _require_string(loaded, "output_dir")
    quant_type = _require_string(loaded, "quant_type")
    compute_dtype = _require_string(loaded, "compute_dtype")
    lr_scheduler_type = _require_string(loaded, "lr_scheduler_type")
    save_strategy = _require_string(loaded, "save_strategy")
    evaluation_strategy = _require_string(loaded, "evaluation_strategy")
    exclude_modules = _require_string(loaded, "exclude_modules")

    use_4bit_quantization = _require_bool(loaded, "use_4bit_quantization")
    if not use_4bit_quantization:
        raise ValueError("use_4bit_quantization must be true for this QLoRA training entrypoint")
    if quant_type != "nf4":
        raise ValueError("quant_type must be nf4 for this QLoRA training entrypoint")
    if compute_dtype != "bfloat16":
        raise ValueError("compute_dtype must be bfloat16 for this QLoRA training entrypoint")
    if save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError("save_strategy must be one of: no, steps, epoch")
    if evaluation_strategy not in {"no", "steps", "epoch"}:
        raise ValueError("evaluation_strategy must be one of: no, steps, epoch")

    warmup_value = loaded["warmup_ratio"]
    if type(warmup_value) not in (int, float) or not 0 <= warmup_value <= 1:
        raise ValueError("warmup_ratio must be a number between 0 and 1")
    warmup_ratio = float(warmup_value)
    lora_dropout_value = loaded["lora_dropout"]
    if type(lora_dropout_value) not in (int, float) or not 0 <= lora_dropout_value < 1:
        raise ValueError("lora_dropout must be a number from 0 (inclusive) to 1 (exclusive)")
    lora_dropout = float(lora_dropout_value)

    target_modules = loaded["target_modules"]
    if not isinstance(target_modules, list) or not target_modules or any(not isinstance(item, str) or not item for item in target_modules):
        raise ValueError("target_modules must be a non-empty list of module names")
    if len(target_modules) != len(set(target_modules)):
        raise ValueError("target_modules must not contain duplicates")

    return TrainingConfig(
        config_path=config_path,
        model=_resolve_project_path(model_name),
        train=_resolve_project_path(dataset_dir) / train_file,
        validation=_resolve_project_path(dataset_dir) / validation_file,
        output=_resolve_project_path(output_dir),
        seed=_require_positive_int(loaded, "seed"),
        use_4bit_quantization=use_4bit_quantization,
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        language_gradient_checkpointing=_require_bool(loaded, "language_gradient_checkpointing"),
        vision_gradient_checkpointing=_require_bool(loaded, "vision_gradient_checkpointing"),
        per_device_train_batch_size=_require_positive_int(loaded, "per_device_train_batch_size"),
        per_device_eval_batch_size=_require_positive_int(loaded, "per_device_eval_batch_size"),
        gradient_accumulation_steps=_require_positive_int(loaded, "gradient_accumulation_steps"),
        num_train_epochs=_require_positive_float(loaded, "num_train_epochs"),
        learning_rate=_require_positive_float(loaded, "learning_rate"),
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_steps=_require_positive_int(loaded, "logging_steps"),
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        save_total_limit=_require_positive_int(loaded, "save_total_limit"),
        lora_rank=_require_positive_int(loaded, "lora_rank"),
        lora_alpha=_require_positive_int(loaded, "lora_alpha"),
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
    )
