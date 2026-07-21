"""Persistent, deployable, and resumable artifacts for one training run."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class RunArtifacts:
    """Own the stable run layout without coupling unit tests to Transformers."""

    def __init__(self, run_dir: Path, config: dict[str, object], manifest: dict[str, Any], app_id: str, run_name: str) -> None:
        self.run_dir = run_dir
        self.config = config
        self.manifest = manifest
        self.app_id = app_id
        self.run_name = run_name
        self.adapters = run_dir / "adapters"
        self.checkpoints = run_dir / "checkpoints"
        self.records = run_dir / "records"
        self.metrics_path = self.records / "metrics.jsonl"
        self.status_path = self.records / "status.json"
        self._best_eval_loss: float | None = None

    def prepare(self, resume: bool) -> Path | None:
        """Initialize a new run or validate a compatible run before resuming."""
        if resume:
            return self._prepare_resume()
        if self.run_dir.exists() and not self._contains_only_launcher_logs():
            raise ValueError(f"run directory already contains artifacts: {self.run_dir}; use --resume or choose a new run_name")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.adapters.mkdir(exist_ok=True)
        self.checkpoints.mkdir(exist_ok=True)
        self.records.mkdir(exist_ok=True)
        _write_json(self.records / "run_config.json", self.config)
        _write_json(self.records / "dataset_manifest.json", self.manifest)
        _write_json(self.records / "run_metadata.json", self.run_metadata())
        self.write_status("running")
        return None

    def _contains_only_launcher_logs(self) -> bool:
        """Allow the wrapper to create the current text log before Python starts."""
        if not self.run_dir.exists():
            return True
        entries = list(self.run_dir.iterdir())
        if not entries:
            return True
        if len(entries) != 1 or entries[0] != self.records or not self.records.is_dir():
            return False
        return all(path.is_file() and path.name.startswith("train_") and path.suffix == ".log" for path in self.records.iterdir())

    def _prepare_resume(self) -> Path:
        config_path = self.records / "run_config.json"
        manifest_path = self.records / "dataset_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"run does not use the resumable artifact layout: {self.run_dir}")
        if json.loads(config_path.read_text(encoding="utf-8")) != self.config:
            raise ValueError("resolved training configuration differs from the frozen run configuration")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
            raise ValueError("dataset manifest differs from the frozen run manifest")
        if not self.checkpoints.is_dir():
            raise ValueError(f"missing checkpoint directory: {self.checkpoints}")
        checkpoints = [
            path for path in self.checkpoints.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-") and path.name.removeprefix("checkpoint-").isdigit()
        ]
        if not checkpoints:
            raise ValueError(f"no checkpoint is available to resume: {self.checkpoints}")
        self.adapters.mkdir(exist_ok=True)
        self.records.mkdir(exist_ok=True)
        self._best_eval_loss = self.best_eval_loss()
        self.write_status("running", resumed_from=str(max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-")))))
        return max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-")))

    def run_metadata(self, **extra: object) -> dict[str, object]:
        return {"app_id": self.app_id, "run_name": self.run_name, "layout_version": 1, **extra}

    def write_status(self, status: str, **extra: object) -> None:
        _write_json(self.status_path, self.run_metadata(status=status, updated_at=_utc_now(), **extra))

    def record_metrics(self, global_step: int, epoch: float | None, metrics: dict[str, Any]) -> None:
        record = {"recorded_at": _utc_now(), "global_step": global_step, "epoch": epoch, "metrics": metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()

    def best_eval_loss(self) -> float | None:
        if not self.metrics_path.is_file():
            return None
        best: float | None = None
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)["metrics"].get("eval_loss")
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if isinstance(value, (int, float)) and (best is None or value < best):
                best = float(value)
        return best

    def should_export_best(self, eval_loss: object) -> bool:
        if not isinstance(eval_loss, (int, float)):
            return False
        value = float(eval_loss)
        if self._best_eval_loss is not None and value >= self._best_eval_loss:
            return False
        self._best_eval_loss = value
        return True

    def export_adapter(self, model: Any, name: str, global_step: int, epoch: float | None, eval_loss: float | None = None) -> None:
        """Publish a complete PEFT adapter directory, retaining the old one on failure."""
        destination = self.adapters / name
        staging = self.adapters / f".{name}.{uuid.uuid4().hex}.tmp"
        previous = self.adapters / f".{name}.previous"
        try:
            model.save_pretrained(staging)
            _write_json(
                staging / "run_metadata.json",
                self.run_metadata(selection=name, global_step=global_step, epoch=epoch, eval_loss=eval_loss),
            )
            if previous.exists():
                shutil.rmtree(previous)
            if destination.exists():
                os.replace(destination, previous)
            os.replace(staging, destination)
            if previous.exists():
                shutil.rmtree(previous)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            if not destination.exists() and previous.exists():
                os.replace(previous, destination)
            raise
