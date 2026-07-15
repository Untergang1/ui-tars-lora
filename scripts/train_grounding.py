#!/usr/bin/env python3
"""QLoRA SFT for UI-TARS/Qwen2.5-VL single-point grounding records."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from training_config import load_training_config
from xformers_vision import enable_xformers_vision_attention


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/lora_qlora.yaml")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved YAML configuration and exit")
    return parser.parse_args()


class GroundingDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not self.rows:
            raise ValueError(f"no examples in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass
class GroundingCollator:
    processor: Any

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError("first-test protocol requires per-device batch size 1")
        item = examples[0]
        user = {"role": "user", "content": [{"type": "image", "image": item["image"]}, {"type": "text", "text": item["prompt"]}]}
        assistant = {"role": "assistant", "content": [{"type": "text", "text": item["response"]}]}
        full_messages = [user, assistant]
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(full_messages)
        batch = self.processor(text=[full_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        prompt_batch = self.processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[:, : int(prompt_batch["attention_mask"][0].sum())] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    try:
        config = load_training_config(args.config)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if args.print_config:
        print(json.dumps(config.as_json(), indent=2))
        return

    if not (config.model / "config.json").is_file():
        raise FileNotFoundError(f"not a local model directory: {config.model}")
    compute_dtype = getattr(torch, config.compute_dtype)
    quantization = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit_quantization,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(config.model, local_files_only=True)
    device_map = "balanced" if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) > 1 else "auto"
    model = AutoModelForVision2Seq.from_pretrained(config.model, quantization_config=quantization, torch_dtype=compute_dtype, device_map=device_map, local_files_only=True)
    model.config.use_cache = False
    patched_vision_blocks = enable_xformers_vision_attention(model)
    print(f"xformers_vision_blocks={patched_vision_blocks}")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=(
            config.language_gradient_checkpointing or config.vision_gradient_checkpointing
        ),
    )
    # The vision encoder is frozen and has no LoRA modules; its checkpointing
    # policy is separate from the language transformer's LoRA backward path.
    model.model.gradient_checkpointing = config.language_gradient_checkpointing
    model.visual.gradient_checkpointing = config.vision_gradient_checkpointing
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.target_modules,
            exclude_modules=config.exclude_modules,
        ),
    )
    adapted_modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    visual_adapters = [name for name in adapted_modules if name.startswith("visual.") or ".visual." in name]
    if visual_adapters:
        raise RuntimeError(f"visual modules must remain frozen, found LoRA adapters: {visual_adapters[:3]}")
    if not adapted_modules:
        raise RuntimeError("no language modules received LoRA adapters")
    print(f"language_lora_modules={len(adapted_modules)}")
    if device_map == "balanced":
        # Keep Trainer in single-process model-parallel mode instead of DDP replication.
        model.is_parallelizable = True
        model.model_parallel = True
    model.print_trainable_parameters()
    config.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(config.output), num_train_epochs=config.num_train_epochs, learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Submodule checkpointing is configured above; do not re-enable it globally.
        gradient_checkpointing=False, bf16=True, logging_steps=config.logging_steps,
        lr_scheduler_type=config.lr_scheduler_type, warmup_ratio=config.warmup_ratio,
        eval_strategy=config.evaluation_strategy, save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit, report_to=[], remove_unused_columns=False, seed=config.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=GroundingDataset(config.train),
        eval_dataset=GroundingDataset(config.validation), data_collator=GroundingCollator(processor),
    )
    trainer.train()
    trainer.save_model(str(config.output))
    processor.save_pretrained(config.output)
    (config.output / "run_config.json").write_text(json.dumps(config.as_json(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
