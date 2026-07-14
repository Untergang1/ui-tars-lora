#!/usr/bin/env python3
"""Score grounding-model responses against the held-out pixel-coordinate labels."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


COORDINATE_PATTERN = re.compile(r"\d+")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=root / "data/processed/validation.jsonl")
    parser.add_argument("--responses", type=Path, required=True, help="JSONL objects with id and response fields")
    parser.add_argument("--report", type=Path, default=root / "outputs/evaluation.json")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    labels = {str(row["id"]): row for row in read_jsonl(args.labels)}
    responses = {str(row["id"]): row for row in read_jsonl(args.responses)}
    results: list[dict[str, object]] = []
    for item_id, label in labels.items():
        response = str(responses.get(item_id, {}).get("response", ""))
        values = [int(value) for value in COORDINATE_PATTERN.findall(response)]
        original = label["original_coordinate"]
        target = label["target_coordinate"]
        width, height = int(original["width"]), int(original["height"])
        if len(values) < 2:
            results.append({"id": item_id, "parseable": False, "response": response})
            continue
        predicted_x = round(values[0] * width / int(target["width"]))
        predicted_y = round(values[1] * height / int(target["height"]))
        distance = math.dist((predicted_x, predicted_y), (int(original["x"]), int(original["y"])))
        diagonal = math.hypot(width, height)
        results.append(
            {
                "id": item_id,
                "parseable": True,
                "response": response,
                "predicted_pixel": {"x": predicted_x, "y": predicted_y},
                "target_pixel": {"x": original["x"], "y": original["y"]},
                "pixel_distance": distance,
                "relative_diagonal_error": distance / diagonal,
            }
        )
    parsed = [row for row in results if row["parseable"]]
    report = {
        "total": len(results),
        "parseable": len(parsed),
        "mean_pixel_distance": sum(float(row["pixel_distance"]) for row in parsed) / len(parsed) if parsed else None,
        "mean_relative_diagonal_error": sum(float(row["relative_diagonal_error"]) for row in parsed) / len(parsed) if parsed else None,
        "within_5_percent_diagonal": sum(float(row["relative_diagonal_error"]) <= 0.05 for row in parsed),
        "examples": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
