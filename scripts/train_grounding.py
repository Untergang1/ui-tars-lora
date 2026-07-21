#!/usr/bin/env python3
"""QLoRA SFT for UI-TARS/Qwen2.5-VL single-point grounding records."""

from __future__ import annotations

import argparse
import json
import os
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments
from grounding_metrics import aggregate, score_label
from run_artifacts import RunArtifacts
from training_config import load_training_config
from xformers_vision import enable_xformers_vision_attention


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/apps/avantage.yaml")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved YAML configuration and exit")
    parser.add_argument("--resume", action="store_true", help="Resume the latest checkpoint in the new run artifact layout")
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
        if not examples:
            raise ValueError("cannot collate an empty batch")
        full_messages = []
        prompt_messages = []
        for item in examples:
            user = {"role": "user", "content": [{"type": "image", "image": item["image"]}, {"type": "text", "text": item["prompt"]}]}
            assistant = {"role": "assistant", "content": [{"type": "text", "text": item["response"]}]}
            full_messages.append([user, assistant])
            prompt_messages.append([user])
        full_text = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) for messages in full_messages]
        prompt_text = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in prompt_messages]
        image_inputs, video_inputs = process_vision_info(full_messages)
        batch = self.processor(text=full_text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        prompt_batch = self.processor(text=prompt_text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[index, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class EpochGenerationEvaluator:
    """Generate and score validation coordinates after completed training epochs."""

    def __init__(self, processor: Any, dataset: GroundingDataset, artifacts: RunArtifacts) -> None:
        self.processor = processor
        self.dataset = dataset
        self.artifacts = artifacts

    @staticmethod
    def completed_epoch(epoch: object) -> int | None:
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
            return None
        rounded = round(float(epoch))
        if rounded <= 0 or abs(float(epoch) - rounded) > 1e-6:
            return None
        return int(rounded)

    def response_for(self, model: Any, item: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["prompt"]},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=32,
            use_cache=True,
        )
        return self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def evaluate(self, model: Any, epoch: int, global_step: int, eval_loss: object) -> tuple[Path, dict[str, object]]:
        was_training = bool(model.training)
        scores: list[dict[str, object]] = []
        try:
            model.eval()
            with torch.inference_mode():
                for item in self.dataset.rows:
                    scores.append(score_label(item, self.response_for(model, item)))
        finally:
            model.train(was_training)

        metrics = aggregate(scores)
        summary = {
            key: metrics[key]
            for key in (
                "total",
                "parseable",
                "parseable_rate",
                "in_contract_range",
                "in_contract_range_rate",
                "bbox_hits",
                "bbox_accuracy",
                "pixel_distance_to_bbox_center",
                "relative_diagonal_error",
            )
        }
        examples = [
            {
                "id": score["id"],
                "bbox_hit": score["bbox_hit"],
                "pixel_distance_to_bbox_center": score.get("pixel_distance_to_bbox_center"),
                "relative_diagonal_error": score.get("relative_diagonal_error"),
            }
            for score in scores
        ]
        loss = float(eval_loss) if isinstance(eval_loss, (int, float)) else None
        path = self.artifacts.record_epoch_evaluation(epoch, global_step, loss, summary, examples)
        return path, summary


class ArtifactCallback(TrainerCallback):
    """Persist metrics and publish adapters independently from Trainer checkpoints."""

    def __init__(self, artifacts: RunArtifacts, generation_evaluator: EpochGenerationEvaluator) -> None:
        self.artifacts = artifacts
        self.generation_evaluator = generation_evaluator

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if logs:
            self.artifacts.record_metrics(state.global_step, state.epoch, logs)
        return control

    def on_evaluate(self, args: TrainingArguments, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        epoch = self.generation_evaluator.completed_epoch(state.epoch)
        if epoch is not None:
            path, generation_metrics = self.generation_evaluator.evaluate(
                kwargs["model"], epoch, state.global_step, metrics.get("eval_loss") if metrics else None
            )
            print(
                "generation_eval "
                f"epoch={epoch} step={state.global_step} path={path} "
                f"bbox_accuracy={generation_metrics['bbox_accuracy']} "
                f"parseable_rate={generation_metrics['parseable_rate']}",
                flush=True,
            )
        if metrics and self.artifacts.should_export_best(metrics.get("eval_loss")):
            self.artifacts.export_adapter(kwargs["model"], "best", state.global_step, state.epoch, float(metrics["eval_loss"]))
        return control

    def on_save(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        self.artifacts.export_adapter(kwargs["model"], "last", state.global_step, state.epoch)
        return control


def build_trainer(config: Any, artifacts: RunArtifacts) -> Trainer:
    if find_spec("tensorboard") is None:
        raise RuntimeError("TensorBoard is required for training records; run scripts/create_environment.sh")
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
    training_args = TrainingArguments(
        output_dir=str(config.checkpoints), logging_dir=str(config.records / "tensorboard"),
        num_train_epochs=config.num_train_epochs, learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Submodule checkpointing is configured above; do not re-enable it globally.
        gradient_checkpointing=False, bf16=True, logging_steps=config.logging_steps,
        lr_scheduler_type=config.lr_scheduler_type, warmup_ratio=config.warmup_ratio,
        eval_strategy=config.evaluation_strategy, save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit, report_to=["tensorboard"], remove_unused_columns=False, seed=config.seed,
    )
    validation_dataset = GroundingDataset(config.validation)
    return Trainer(
        model=model, args=training_args, train_dataset=GroundingDataset(config.train),
        eval_dataset=validation_dataset, data_collator=GroundingCollator(processor),
        callbacks=[ArtifactCallback(artifacts, EpochGenerationEvaluator(processor, validation_dataset, artifacts))],
    )


def main() -> None:
    args = parse_args()
    try:
        config = load_training_config(args.config)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if args.print_config:
        print(json.dumps(config.as_json(), indent=2))
        return

    if not config.train.is_file() or not config.validation.is_file():
        raise FileNotFoundError(f"processed dataset is incomplete for {config.app_id}; run prepare_grounding_data.py first")
    if not config.manifest.is_file():
        raise FileNotFoundError(f"missing dataset manifest for {config.app_id}; run prepare_grounding_data.py first")
    manifest = json.loads(config.manifest.read_text(encoding="utf-8"))
    if manifest.get("app_id") != config.app_id:
        raise ValueError(f"dataset manifest app_id does not match configuration: {config.app_id}")
    artifacts = RunArtifacts(config.output, config.as_json(), manifest, config.app_id, config.run_name)
    resume_checkpoint = artifacts.prepare(args.resume)
    try:
        trainer = build_trainer(config, artifacts)
    except KeyboardInterrupt:
        artifacts.write_status("interrupted", global_step=0, epoch=None, best_eval_loss=artifacts.best_eval_loss())
        raise
    except BaseException as error:
        artifacts.write_status("failed", global_step=0, epoch=None, best_eval_loss=artifacts.best_eval_loss(), error_type=type(error).__name__)
        raise
    try:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    except KeyboardInterrupt:
        export_error: str | None = None
        try:
            artifacts.export_adapter(trainer.model, "last", trainer.state.global_step, trainer.state.epoch)
        except Exception as error:
            export_error = type(error).__name__
        artifacts.write_status("interrupted", global_step=trainer.state.global_step, epoch=trainer.state.epoch,
                               best_eval_loss=artifacts.best_eval_loss(), adapter_export_error=export_error)
        raise
    except BaseException as error:
        artifacts.write_status("failed", global_step=trainer.state.global_step, epoch=trainer.state.epoch,
                               best_eval_loss=artifacts.best_eval_loss(), error_type=type(error).__name__)
        raise
    artifacts.export_adapter(trainer.model, "last", trainer.state.global_step, trainer.state.epoch)
    artifacts.write_status("completed", global_step=trainer.state.global_step, epoch=trainer.state.epoch,
                           best_eval_loss=artifacts.best_eval_loss())


if __name__ == "__main__":
    main()
