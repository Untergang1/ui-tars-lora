"""Tests for application-isolated bbox label preparation."""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_grounding_data import load_contract, load_rows, make_training_records, stable_split  # noqa: E402
from training_config import load_training_config  # noqa: E402


class GroundingDataTests(unittest.TestCase):
    def write_config(self, root: Path) -> Path:
        values = yaml.safe_load((PROJECT_ROOT / "configs/apps/avantage.yaml").read_text(encoding="utf-8"))
        values.update({"data_root": str(root), "app_id": "test-app", "run_name": "test"})
        path = root / "test-app.yaml"
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return path

    def write_annotations(self, root: Path) -> None:
        images = root / "images"
        images.mkdir()
        Image.new("RGB", (10, 10)).save(images / "shared.png")
        with (root / "annotations.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("id", "image", "description", "left", "top", "right", "bottom", "app_version", "theme"))
            writer.writerow(("first", "shared.png", "first target", 2, 2, 6, 6, "1.0", "light"))
            writer.writerow(("second", "shared.png", "second target", 3, 3, 5, 5, "1.0", "dark"))
            writer.writerow(("third", "shared.png", "third target", 0, 0, 1, 1, "1.1", "light"))

    def test_bbox_rows_create_center_targets_and_deterministic_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "test-app"
            root.mkdir()
            self.write_annotations(root)
            config = load_training_config(self.write_config(root))
            rows = load_rows(config)
            train, validation = stable_split(rows, config.split_seed, config.validation_fraction)
            again_train, again_validation = stable_split(rows, config.split_seed, config.validation_fraction)
            records = make_training_records(train, config, load_contract(PROJECT_ROOT / "configs/grounding_contract.json"))

            self.assertEqual(len(train), 2)
            self.assertEqual(len(validation), 1)
            self.assertEqual([row["id"] for row in train], [row["id"] for row in again_train])
            self.assertEqual([row["id"] for row in validation], [row["id"] for row in again_validation])
            self.assertTrue(all(record["app_id"] == "test-app" for record in records))
            first = next(record for record in records if record["id"] == "first")
            self.assertEqual(first["bbox_center"], {"x": 3.5, "y": 3.5})
            self.assertEqual(first["target_coordinate"]["x"], 672)

    def test_invalid_bbox_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "test-app"
            root.mkdir()
            self.write_annotations(root)
            csv_path = root / "annotations.csv"
            content = csv_path.read_text(encoding="utf-8").replace("0,0,1,1,1.1", "0,0,0,1,1.1")
            csv_path.write_text(content, encoding="utf-8")
            config = load_training_config(self.write_config(root))
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "1"):
                load_rows(config)


if __name__ == "__main__":
    unittest.main()
