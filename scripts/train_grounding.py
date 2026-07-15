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
from xformers_vision import enable_xformers_vision_attention


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXCLUDED_MODULES = r"^visual\..*"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "models/UI-TARS-1.5-7B")
    parser.add_argument("--train", type=Path, default=root / "data/processed/train.jsonl")
    parser.add_argument("--validation", type=Path, default=root / "data/processed/validation.jsonl")
    parser.add_argument("--output", type=Path, default=root / "outputs/avantage-grounding-qlora")
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260714)
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
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"not a local model directory: {args.model}")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    device_map = "balanced" if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) > 1 else "auto"
    model = AutoModelForVision2Seq.from_pretrained(args.model, quantization_config=quantization, torch_dtype=torch.bfloat16, device_map=device_map, local_files_only=True)
    model.config.use_cache = False
    patched_vision_blocks = enable_xformers_vision_attention(model)
    print(f"xformers_vision_blocks={patched_vision_blocks}")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # Vision encoder is frozen; checkpointing it needlessly retains multi-gigabyte attention activations.
    model.visual.gradient_checkpointing = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES, exclude_modules=EXCLUDED_MODULES))
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
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=False, bf16=True, logging_steps=1, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, report_to=[], remove_unused_columns=False, seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=GroundingDataset(args.train),
        eval_dataset=GroundingDataset(args.validation), data_collator=GroundingCollator(processor),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    processor.save_pretrained(args.output)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
