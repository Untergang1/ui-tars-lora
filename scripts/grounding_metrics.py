"""Shared parsing and scoring for original-pixel grounding responses."""

from __future__ import annotations

import math
import re
from typing import Iterable


COORDINATE_PATTERN = re.compile(r"\d+")


def score_label(label: dict[str, object], response: object | None) -> dict[str, object]:
    """Score one response using the grounding contract's coordinate parser."""
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
    }
    values = [int(value) for value in COORDINATE_PATTERN.findall(text)]
    if len(values) < 2:
        result["failure_reason"] = "unparseable_response"
        return result
    target = label["target_coordinate"]
    original = label["original_image"]
    bbox = label["bbox"]
    assert isinstance(target, dict) and isinstance(original, dict) and isinstance(bbox, dict)
    width, height = int(original["width"]), int(original["height"])
    if int(target["width"]) != width or int(target["height"]) != height:
        raise ValueError(f"label {item_id} does not use original-image coordinate dimensions")
    predicted_model = {"x": values[0], "y": values[1]}
    result["parseable"] = True
    result["predicted_model_coordinate"] = predicted_model
    if not (0 <= values[0] < width and 0 <= values[1] < height):
        result["failure_reason"] = "coordinate_outside_contract"
        return result
    result["in_contract_range"] = True
    predicted_x, predicted_y = values[0], values[1]
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


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate response scores with all accuracy rates using the full denominator."""
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
        "in_contract_range_rate": len(in_range) / len(rows) if rows else None,
        "bbox_hits": len(hits),
        "bbox_accuracy": len(hits) / len(rows) if rows else None,
        "bbox_accuracy_among_parseable": len(hits) / len(parseable) if parseable else None,
        "mean_pixel_distance_to_bbox_center": sum(distances) / len(distances) if distances else None,
        "mean_relative_diagonal_error": sum(relative_errors) / len(relative_errors) if relative_errors else None,
        "pixel_distance_to_bbox_center": _distribution(distances),
        "relative_diagonal_error": _distribution(relative_errors),
    }
