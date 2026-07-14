#!/usr/bin/env python3
"""Validate pixel-coordinate labels and create a deterministic 16/4 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


REQUIRED_COLUMNS = ("id", "image", "description", "x", "y")


def error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=root / "data/annotations/grounding.csv")
    parser.add_argument("--images-dir", type=Path, default=root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--contract", type=Path, default=root / "configs/grounding_contract.json")
    parser.add_argument("--seed", default="20260714", help="Stable split seed, recorded in manifest")
    parser.add_argument("--validation-size", type=int, default=4)
    return parser.parse_args()


def load_rows(annotations: Path, images_dir: Path) -> list[dict[str, object]]:
    if not annotations.is_file():
        error(f"annotations file does not exist: {annotations}")
    with annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            error("CSV header must be exactly: id,image,description,x,y")
        rows = list(reader)

    if len(rows) != 20:
        error(f"expected exactly 20 labels for this first test, got {len(rows)}")

    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    validated: list[dict[str, object]] = []
    for row in rows:
        item_id = row["id"].strip()
        image_name = row["image"].strip()
        description = row["description"].strip()
        if not item_id or not image_name or not description:
            error("id, image, and description cannot be empty")
        if item_id in seen_ids:
            error(f"duplicate id: {item_id}")
        if image_name in seen_images:
            error(f"duplicate image: {image_name}")
        image_path = (images_dir / image_name).resolve()
        if images_dir.resolve() not in image_path.parents or not image_path.is_file():
            error(f"image must exist inside {images_dir}: {image_name}")
        try:
            x, y = int(row["x"]), int(row["y"])
        except ValueError:
            error(f"coordinates must be integers for {item_id}")
        with Image.open(image_path) as image:
            width, height = image.size
        if not (0 <= x < width and 0 <= y < height):
            error(f"coordinate ({x}, {y}) is outside {image_name} ({width}x{height})")
        seen_ids.add(item_id)
        seen_images.add(image_name)
        validated.append(
            {
                "id": item_id,
                "image": image_name,
                "image_path": str(image_path),
                "description": description,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
    return validated


def stable_split(rows: list[dict[str, object]], seed: str, validation_size: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if validation_size != 4:
        error("the first-test protocol requires exactly 4 validation examples")
    ranked = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    validation_ids = {row["id"] for row in ranked[:validation_size]}
    train = [row for row in rows if row["id"] not in validation_ids]
    validation = [row for row in rows if row["id"] in validation_ids]
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_contract(path: Path) -> dict[str, object]:
    if not path.is_file():
        error(f"grounding contract does not exist: {path}; run extract_agent_s_contract.py first")
    contract = json.loads(path.read_text(encoding="utf-8"))
    coordinate_space = contract.get("coordinate_space", {})
    if coordinate_space != {"width": 1920, "height": 1080}:
        error("unexpected coordinate space; review the Agent-S contract before preparing data")
    if contract.get("prompt_template") != "Query:{description}\nOutput only the coordinate of one point in your response.\n":
        error("unexpected Agent-S prompt contract; review before preparing data")
    return contract


def make_training_records(rows: list[dict[str, object]], contract: dict[str, object]) -> list[dict[str, object]]:
    coordinate_space = contract["coordinate_space"]
    target_width = int(coordinate_space["width"])
    target_height = int(coordinate_space["height"])
    template = str(contract["prompt_template"])
    response_template = str(contract["assistant_response_template"])
    records: list[dict[str, object]] = []
    for row in rows:
        model_x = round(int(row["x"]) * target_width / int(row["width"]))
        model_y = round(int(row["y"]) * target_height / int(row["height"]))
        records.append(
            {
                "id": row["id"],
                "image": row["image_path"],
                "prompt": template.format(description=row["description"]),
                "response": response_template.format(x=model_x, y=model_y),
                "target_coordinate": {"x": model_x, "y": model_y, "width": target_width, "height": target_height},
                "original_coordinate": {"x": row["x"], "y": row["y"], "width": row["width"], "height": row["height"]},
            }
        )
    return records


def main() -> None:
    args = parse_args()
    rows = load_rows(args.annotations, args.images_dir)
    contract = load_contract(args.contract)
    train, validation = stable_split(rows, args.seed, args.validation_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_grounding.jsonl", train)
    write_jsonl(args.output_dir / "validation_grounding.jsonl", validation)
    write_jsonl(args.output_dir / "train.jsonl", make_training_records(train, contract))
    write_jsonl(args.output_dir / "validation.jsonl", make_training_records(validation, contract))
    manifest = {
        "annotation_file": str(args.annotations.resolve()),
        "images_dir": str(args.images_dir.resolve()),
        "seed": args.seed,
        "train_count": len(train),
        "validation_count": len(validation),
        "validation_ids": [row["id"] for row in validation],
        "coordinate_origin": "top-left",
        "coordinate_unit": "original screenshot pixels",
        "agent_s_contract": str(args.contract.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(train)} train and {len(validation)} validation labels to {args.output_dir}")


if __name__ == "__main__":
    main()
