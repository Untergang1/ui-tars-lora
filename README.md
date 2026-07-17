# UI-TARS Per-Application Grounding LoRA

This project fine-tunes the local `ByteDance-Seed/UI-TARS-1.5-7B` model for
UI grounding while keeping one LoRA adapter per application:

```text
screenshot + UI element description -> Agent-S-compatible coordinate response
```

The Agent-S contract remains a single coordinate response. Labels include a
full bounding box: its geometric center supplies the single-point training
target, and validation considers a prediction correct when it falls inside the
box. The production vLLM service remains on GPU 2 and port 18000; training uses
GPU 0 and temporary adapter evaluation uses GPU 1.

## Application Layout

Every application has a separate ignored data root:

```text
data/<app_id>/
  images/                 # sensitive PNG/JPEG screenshots
  annotations.csv         # sensitive bbox labels
  processed/              # generated JSONL and manifest
outputs/<app_id>/<run_name>/
  adapter files and frozen run/data metadata
configs/apps/<app_id>.yaml
```

`<app_id>` is a stable lowercase slug such as `avantage` or `omnic`. The
application YAML is the single source of truth for data paths, split settings,
training parameters, and output isolation. The checked-in Avantage and Omnic
profiles are starting points; copy `configs/apps/app.template.yaml` for a new
application.

## BBox Annotation Format

Create `data/<app_id>/annotations.csv` with this exact header:

```csv
id,image,description,left,top,right,bottom,app_version,theme
toolbar-acquire-001,main-window.png,The "Acquire" button in the spectrum toolbar,1100,160,1184,210,8.2,light
```

- `id` is unique within the application and remains stable as data grows.
- `image` is a path relative to the application's `images/` directory.
- `left,top,right,bottom` are integer original-screenshot pixels. Left/top are
  inclusive; right/bottom are exclusive. A valid box satisfies
  `0 <= left < right <= image_width` and `0 <= top < bottom <= image_height`.
- `app_version` and `theme` are retained in the evaluation report for regression
  analysis. Blank or whitespace-only values are normalized to `unknown`.
- The training point is the geometric center of the pixel box:
  `((left + right - 1) / 2, (top + bottom - 1) / 2)`.

Screenshots and CSV files are sensitive and ignored by Git. Remove account
names, identifiers, paths, and other retained material before annotation.

## Workflow

1. Run `scripts/check_runtime.sh` and `scripts/copy_model.sh`.
2. Pin the Agent-S prompt with `scripts/pin_sources.sh` and
   `scripts/extract_agent_s_contract.py`.
3. Create the application image directory and CSV, then prepare it:

   ```bash
   python3 scripts/prepare_grounding_data.py --config configs/apps/avantage.yaml
   ```

   It accepts any data size of at least two labels, deterministically assigns
   80%/20% train/validation label rows, and records input hashes and split IDs
   in `processed/manifest.json`. It warns when the same screenshot appears in
   both splits; this is expected with row-level splitting but can inflate
   validation results.

4. Create the isolated environment with `scripts/create_environment.sh`, then
   train one application adapter on GPU 0:

   ```bash
   scripts/run_train.sh configs/apps/avantage.yaml
   ```

   Change `run_name` in the application YAML before a distinct experiment. The
   resulting `outputs/<app_id>/<run_name>/` freezes the resolved configuration,
   dataset manifest, and application metadata beside the adapter.

5. Start an adapter only with an explicit application identity on GPU 1:

   ```bash
   scripts/serve_adapter.sh --app avantage \
     --adapter "$(pwd)/outputs/avantage/baseline"
   ```

   The script rejects port 18000 and refuses an adapter whose metadata belongs
   to a different application.

6. Evaluate JSONL responses containing `id` and `response`:

   ```bash
   python3 scripts/evaluate_grounding.py \
     --config configs/apps/avantage.yaml \
     --responses outputs/avantage/baseline/responses.jsonl
   ```

   The report defaults to `outputs/<app_id>/<run_name>/evaluation.json` and
   includes total-denominator bbox accuracy, parseability, out-of-contract
   responses, center-distance diagnostics, per-example results, and grouped
   metrics by application version and theme.

## Validation

Validate a profile without loading model weights or data:

```bash
/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora/bin/python \
  scripts/train_grounding.py --config configs/apps/avantage.yaml --print-config
```

Run the repository tests with `python3 -m unittest discover -s tests`.
