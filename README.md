# UI-TARS-1.5 Avantage Grounding LoRA

This project tests whether the Qwen2.5-VL LoRA/SFT workflow can fine-tune the
locally deployed `ByteDance-Seed/UI-TARS-1.5-7B` model for UI grounding:

```text
screenshot + UI element description -> Agent-S-compatible coordinate response
```

The production vLLM service is intentionally out of scope. Training uses a
project-local copy of its cached base model, a separate conda environment, and
GPU 0. The production service remains on GPU 2 and port 18000.

## Layout

- `models/UI-TARS-1.5-7B`: read-only local model copy created by
  `scripts/copy_model.sh` (ignored by Git).
- `third_party/Qwen2.5-VL`: shallow clone of the upstream Qwen repository.
- `third_party/Agent-S`: shallow clone used to pin the exact grounding prompt.
- `data/raw`: the 20 supplied screenshots (ignored by Git).
- `data/annotations/grounding.csv`: coordinate labels (ignored by Git).
- `data/processed`: validated train/validation JSONL and manifests.
- `outputs`: LoRA adapters, logs, and evaluation reports.

## Screenshot Handoff

Place PNG/JPEG screenshots under `data/raw/` and create
`data/annotations/grounding.csv` with exactly these columns:

```csv
id,image,description,x,y
avantage_001,avantage_001.png,The "Acquire" button in the spectrum toolbar,1142,186
```

- `id` must be unique and stable.
- `image` is relative to `data/raw/`.
- `description` is the grounding query supplied to the model.
- `x,y` are the known correct pixel coordinates in the original, unmodified
  screenshot, with origin at the upper-left.
- Supply 20 distinct screenshots. Use the deterministic 16/4 split generated
  by `scripts/prepare_grounding_data.py`; do not place near-duplicate states in
  both sets.

Remove account names, sample identifiers, project paths, and other material
that should not be retained in the training data before adding screenshots.

## Workflow

1. Run `scripts/check_runtime.sh` and `scripts/copy_model.sh`.
2. Once source clones complete, run `scripts/pin_sources.sh` and
   `scripts/extract_agent_s_contract.py`.
3. Create the CSV and images above, then run
   `scripts/prepare_grounding_data.py`.
4. Create the isolated environment with `scripts/create_environment.sh`.
5. Run `scripts/run_train.sh` to launch the Qwen2.5-VL-compatible QLoRA
   baseline on GPU 0. It uses only the local UI-TARS copy and does not contact
   the network. The public Qwen repository is optional provenance because this
   host cannot reach GitHub.
6. Evaluate the adapter using a temporary vLLM process on GPU 1 and a local
   loopback port. Never target the production port 18000.

No Git commit is made by these scripts. Set `user.name` and `user.email`
locally before committing any project changes.
