#!/usr/bin/env python3
"""Validate one application's bbox labels and create deterministic JSONL splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

from training_config import TrainingConfig, load_training_config


REQUIRED_COLUMNS = (
    "id",
    "image",
    "description",
    "left",
    "top",
    "right",
    "bottom",
    "app_version",
)


def error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/apps/avantage.yaml")
    parser.add_argument("--contract", type=Path, default=root / "configs/grounding_contract.json")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(config: TrainingConfig) -> list[dict[str, object]]:
    if not config.annotations.is_file():
        error(f"annotations file does not exist: {config.annotations}")
    if not config.images.is_dir():
        error(f"images directory does not exist: {config.images}")
    with config.annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            error(f"CSV header must be exactly: {','.join(REQUIRED_COLUMNS)}")
        rows = list(reader)

    if len(rows) < 2:
        error("at least two labels are required to create train and validation splits")
    images_root = config.images.resolve()
    seen_ids: set[str] = set()
    validated: list[dict[str, object]] = []
    for row in rows:
        item_id = row["id"].strip()
        image_name = row["image"].strip()
        description = row["description"].strip()
        app_version = row["app_version"].strip() or "unknown"
        if not item_id or not image_name or not description:
            error("id, image, and description cannot be empty")
        if item_id in seen_ids:
            error(f"duplicate id: {item_id}")
        image_path = (images_root / image_name).resolve()
        if images_root not in image_path.parents or not image_path.is_file():
            error(f"image must exist inside {config.images}: {image_name}")
        try:
            left, top, right, bottom = (int(row[key]) for key in ("left", "top", "right", "bottom"))
        except ValueError:
            error(f"bbox coordinates must be integers for {item_id}")
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            error(f"cannot read image for {item_id}: {exc}")
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            error(f"bbox ({left}, {top}, {right}, {bottom}) is outside {image_name} ({width}x{height})")
        seen_ids.add(item_id)
        validated.append(
            {
                "id": item_id,
                "image": image_name,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "description": description,
                "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                "app_version": app_version,
                "width": width,
                "height": height,
            }
        )
    return validated


def stable_split(rows: list[dict[str, object]], seed: str, validation_fraction: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validation_count = min(len(rows) - 1, max(1, round(len(rows) * validation_fraction)))
    ranked = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    validation_ids = {str(row["id"]) for row in ranked[:validation_count]}
    train = [row for row in rows if str(row["id"]) not in validation_ids]
    validation = [row for row in rows if str(row["id"]) in validation_ids]
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_contract(path: Path) -> dict[str, object]:
    if not path.is_file():
        error(f"grounding contract does not exist: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("coordinate_space") != {"width": 1920, "height": 1080}:
        error("unexpected coordinate space in grounding contract")
    if contract.get("prompt_template") != "Query:{description}\nOutput only the coordinate of one point in your response.\n":
        error("unexpected prompt template in grounding contract")
    return contract


def make_training_records(rows: list[dict[str, object]], config: TrainingConfig, contract: dict[str, object]) -> list[dict[str, object]]:
    coordinate_space = contract["coordinate_space"]
    target_width = int(coordinate_space["width"])
    target_height = int(coordinate_space["height"])
    template = str(contract["prompt_template"])
    response_template = str(contract["assistant_response_template"])
    records: list[dict[str, object]] = []
    for row in rows:
        bbox = row["bbox"]
        assert isinstance(bbox, dict)
        center_x = (int(bbox["left"]) + int(bbox["right"]) - 1) / 2
        center_y = (int(bbox["top"]) + int(bbox["bottom"]) - 1) / 2
        model_x = round(center_x * target_width / int(row["width"]))
        model_y = round(center_y * target_height / int(row["height"]))
        records.append(
            {
                "id": row["id"],
                "app_id": config.app_id,
                "image": row["image_path"],
                "image_name": row["image"],
                "image_sha256": row["image_sha256"],
                "prompt": template.format(description=row["description"]),
                "response": response_template.format(x=model_x, y=model_y),
                "target_coordinate": {"x": model_x, "y": model_y, "width": target_width, "height": target_height},
                "bbox": bbox,
                "bbox_center": {"x": center_x, "y": center_y},
                "original_image": {"width": row["width"], "height": row["height"]},
                "app_version": row["app_version"],
            }
        )
    return records


def image_overlap(train: list[dict[str, object]], validation: list[dict[str, object]]) -> list[str]:
    train_images = {str(row["image"]) for row in train}
    validation_images = {str(row["image"]) for row in validation}
    return sorted(train_images & validation_images)


def main() -> None:
    args = parse_args()
    try:
        config = load_training_config(args.config)
    except ValueError as exc:
        error(str(exc))
    rows = load_rows(config)
    contract = load_contract(args.contract)
    train, validation = stable_split(rows, config.split_seed, config.validation_fraction)
    config.processed.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.processed / "train_grounding.jsonl", train)
    write_jsonl(config.processed / "validation_grounding.jsonl", validation)
    write_jsonl(config.train, make_training_records(train, config, contract))
    write_jsonl(config.validation, make_training_records(validation, config, contract))
    overlap = image_overlap(train, validation)
    image_inventory = {
        str(row["image"]): {
            "sha256": row["image_sha256"],
            "width": row["width"],
            "height": row["height"],
        }
        for row in sorted(rows, key=lambda item: str(item["image"]))
    }
    manifest = {
        "app_id": config.app_id,
        "annotation_file": str(config.annotations.resolve()),
        "annotation_sha256": sha256_file(config.annotations),
        "images_dir": str(config.images.resolve()),
        "split_seed": config.split_seed,
        "validation_fraction": config.validation_fraction,
        "train_count": len(train),
        "validation_count": len(validation),
        "validation_ids": [row["id"] for row in validation],
        "coordinate_origin": "top-left",
        "coordinate_unit": "original screenshot pixels",
        "bbox_convention": "left/top inclusive; right/bottom exclusive",
        "target_point": "bbox geometric center",
        "grounding_contract": str(args.contract.resolve()),
        "grounding_contract_sha256": sha256_file(args.contract),
        "cross_split_images": overlap,
        "cross_split_image_count": len(overlap),
        "images": image_inventory,
    }
    (config.processed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(train)} train and {len(validation)} validation labels for {config.app_id} to {config.processed}")
    if overlap:
        print(f"warning: {len(overlap)} image(s) occur in both splits because splitting is by label row", file=sys.stderr)


if __name__ == "__main__":
    main()
