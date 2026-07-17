#!/usr/bin/env python3
"""Score one application's point responses against held-out bbox labels."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from training_config import load_training_config


COORDINATE_PATTERN = re.compile(r"\d+")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/apps/avantage.yaml")
    parser.add_argument("--responses", type=Path, required=True, help="JSONL objects with id and response fields")
    parser.add_argument("--report", type=Path, help="Defaults to the configured application run directory")
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


def score_label(label: dict[str, object], response: object | None) -> dict[str, object]:
    item_id = str(label["id"])
    text = "" if response is None else str(response)
    result: dict[str, object] = {
        "id": item_id,
        "response": text,
        "parseable": False,
        "in_contract_range": False,
        "bbox_hit": False,
        "failure_reason": None,
        "app_version": label["app_version"],
        "description_uia_referenced": label["description_uia_referenced"],
    }
    values = [int(value) for value in COORDINATE_PATTERN.findall(text)]
    if len(values) < 2:
        result["failure_reason"] = "unparseable_response"
        return result
    target = label["target_coordinate"]
    original = label["original_image"]
    bbox = label["bbox"]
    assert isinstance(target, dict) and isinstance(original, dict) and isinstance(bbox, dict)
    predicted_model = {"x": values[0], "y": values[1]}
    result["parseable"] = True
    result["predicted_model_coordinate"] = predicted_model
    target_width, target_height = int(target["width"]), int(target["height"])
    if not (0 <= values[0] <= target_width and 0 <= values[1] <= target_height):
        result["failure_reason"] = "coordinate_outside_contract"
        return result
    result["in_contract_range"] = True
    width, height = int(original["width"]), int(original["height"])
    predicted_x = round(values[0] * width / target_width)
    predicted_y = round(values[1] * height / target_height)
    result["predicted_pixel"] = {"x": predicted_x, "y": predicted_y}
    result["target_bbox"] = bbox
    center = label["bbox_center"]
    assert isinstance(center, dict)
    distance = math.dist((predicted_x, predicted_y), (float(center["x"]), float(center["y"])))
    result["bbox_center_pixel"] = center
    result["pixel_distance_to_bbox_center"] = distance
    result["relative_diagonal_error"] = distance / math.hypot(width, height)
    hit = (
        int(bbox["left"]) <= predicted_x < int(bbox["right"])
        and int(bbox["top"]) <= predicted_y < int(bbox["bottom"])
    )
    result["bbox_hit"] = hit
    if not hit:
        result["failure_reason"] = "point_outside_bbox"
    return result


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    parseable = [row for row in rows if row["parseable"]]
    in_range = [row for row in rows if row["in_contract_range"]]
    hits = [row for row in rows if row["bbox_hit"]]
    distances = [float(row["pixel_distance_to_bbox_center"]) for row in in_range]
    relative_errors = [float(row["relative_diagonal_error"]) for row in in_range]
    return {
        "total": len(rows),
        "parseable": len(parseable),
        "parseable_rate": len(parseable) / len(rows) if rows else None,
        "in_contract_range": len(in_range),
        "bbox_hits": len(hits),
        "bbox_accuracy": len(hits) / len(rows) if rows else None,
        "bbox_accuracy_among_parseable": len(hits) / len(parseable) if parseable else None,
        "mean_pixel_distance_to_bbox_center": sum(distances) / len(distances) if distances else None,
        "mean_relative_diagonal_error": sum(relative_errors) / len(relative_errors) if relative_errors else None,
    }


def grouped_aggregate(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {value: aggregate(group) for value, group in sorted(groups.items())}


def main() -> None:
    args = parse_args()
    try:
        config = load_training_config(args.config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if not config.validation.is_file() or not config.manifest.is_file():
        raise SystemExit(f"error: processed validation data is missing for {config.app_id}; run prepare_grounding_data.py first")
    manifest = json.loads(config.manifest.read_text(encoding="utf-8"))
    if manifest.get("app_id") != config.app_id:
        raise SystemExit(f"error: dataset manifest app_id does not match configuration: {config.app_id}")
    labels = read_jsonl(config.validation)
    if any(str(label.get("app_id", "")) != config.app_id for label in labels):
        raise SystemExit(f"error: validation labels do not all belong to {config.app_id}")
    try:
        responses = read_responses(args.responses)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: invalid responses: {exc}") from exc
    results = [score_label(label, responses.get(str(label["id"]))) for label in labels]
    label_ids = {str(label["id"]) for label in labels}
    report = {
        "app_id": config.app_id,
        "dataset_manifest": str(config.manifest),
        "bbox_convention": "left/top inclusive; right/bottom exclusive",
        "metrics": aggregate(results),
        "by_app_version": grouped_aggregate(results, "app_version"),
        "by_description_uia_referenced": grouped_aggregate(results, "description_uia_referenced"),
        "unexpected_response_ids": sorted(set(responses) - label_ids),
        "examples": results,
    }
    report_path = args.report or config.output / "evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"app_id": config.app_id, **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
