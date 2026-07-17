# Repository Guidelines

## Scope and Priorities

- These instructions apply to the entire repository unless a nested `AGENTS.md`
  provides more-specific rules.
- Preserve existing user changes. Do not revert, stage, or commit work that was
  already present when you began a task unless the user explicitly asks.
- Keep changes small and focused. Update documentation and configuration when a
  behavior or workflow change requires it.

## Project Overview

This repository fine-tunes the local `ByteDance-Seed/UI-TARS-1.5-7B` model for
UI grounding: a screenshot and element description produce an Agent-S-compatible
coordinate response. The production vLLM service is out of scope and must remain
isolated on GPU 2 and port 18000.

- `configs/` defines the grounding contract and LoRA/QLoRA parameters.
- `scripts/` contains the reproducible data, environment, training, and
  evaluation entrypoints.
- `data/<app_id>/images/` and `annotations.csv` are user-provided,
  ignored source material. Its `processed/` directory, `models/`, `outputs/`,
  and `logs/` are generated or large local assets and must remain untracked.
- `third_party/` contains ignored upstream clones; record their pinned revisions
  in `SOURCES.md`, not the vendor directories.

## Development Workflow

1. Read the relevant script, configuration, and documentation before changing a
   workflow. Follow the paths and defaults already established by the scripts.
2. For training-related work, run `scripts/check_runtime.sh` first. It protects
   the production UI-TARS service and verifies the required storage headroom.
3. Use the intended sequence: `scripts/copy_model.sh`,
   `scripts/pin_sources.sh`, `scripts/extract_agent_s_contract.py`,
   `scripts/prepare_grounding_data.py --config configs/apps/<app_id>.yaml`,
   `scripts/create_environment.sh`, and `scripts/run_train.sh` as applicable.
   Training must use GPU 0; evaluation uses a temporary service on GPU 1 and
   never port 18000.
4. Do not access the network, alter the production service, or delete local
   models, data, adapters, or outputs unless the user explicitly requests it.

## Data and Contract Rules

- Treat screenshots and annotation CSV files as sensitive. Do not add them to
  Git or expose account names, identifiers, paths, or other retained material.
- Each application has its own lowercase-slug directory, YAML profile, processed
  data, and LoRA output. The caller must explicitly select the application;
  never infer it from a screenshot or mix application records.
- The CSV header is exactly
  `id,image,description,left,top,right,bottom,app_version`. IDs are unique
  within an application; image rows may repeat. Blank `app_version` values are
  normalized to `unknown`.
- Bboxes use integer original-screenshot pixels with top-left origin: left/top
  are inclusive and right/bottom are exclusive. The training point is the bbox
  geometric center. Preserve the 1920x1080 Agent-S single-point prompt and
  response contract unless all dependent code changes together.
- Keep data splitting deterministic. The configured 80/20 split is by label
  row, not distinct screenshot; record and surface any cross-split image reuse.

## Code and Validation

- Follow the existing Python and Bash style: Python uses type annotations and
  `pathlib`; Bash scripts use `#!/usr/bin/env bash` and `set -euo pipefail`.
- Prefer standard-library solutions. Add dependencies only when necessary and
  document why they are needed.
- Run the narrowest relevant validation after a change. At minimum, run
  `python3 -m py_compile` on changed Python files and `bash -n` on changed Bash
  scripts. Run the affected workflow command when its required local inputs are
  available; do not fabricate sensitive data just to run it.
- Before handoff, inspect `git diff --check`, the final diff, and `git status`.

## Git Requirements

- After completing every modification to Git-trackable files, automatically
  create a Git commit before handing off. This requirement supersedes older
  documentation that says scripts or agents do not commit changes.
- Stage only the files changed for the current task. Never include pre-existing
  user changes, ignored assets, generated artifacts, credentials, screenshots,
  annotation CSV files, model weights, adapters, or third-party clones.
- Use a concise imperative commit message that describes the completed change,
  for example `docs: add repository agent guidance`.
- If Git cannot commit because identity or another repository condition is
  missing, report the exact blocker and leave the task files intact and staged
  only when doing so is safe. Do not modify user-level Git configuration.
