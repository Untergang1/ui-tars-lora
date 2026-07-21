# UI-TARS Per-Application Grounding LoRA

This project fine-tunes the local `ByteDance-Seed/UI-TARS-1.5-7B` model for
UI grounding while keeping one LoRA adapter per application:

```text
screenshot + UI element description -> single-coordinate response
```

The grounding contract uses a single coordinate response. Labels include a full
bounding box: its geometric center supplies the single-point training target,
and validation considers a prediction correct when it falls inside the box. The
production vLLM service remains on GPU 2 and port 18000; training uses GPU 0
and temporary adapter evaluation uses GPU 1.

## Application Layout

Every application has a separate ignored data root:

```text
data/<app_id>/
  images/                 # sensitive PNG/JPEG screenshots
  annotations.csv         # sensitive bbox labels
  processed/              # generated JSONL and manifest
outputs/<app_id>/<run_name>/
  adapters/
    last/               # latest deployable PEFT adapter
    best/               # lowest validation-loss deployable adapter
  checkpoints/          # resumable Trainer state, retained per save_total_limit
  records/              # frozen config/data, JSONL metrics, text logs, TensorBoard events
configs/apps/<app_id>.yaml
```

`<app_id>` is a stable lowercase slug such as `avantage` or `omnic`. The
application YAML is the single source of truth for data paths, split settings,
training parameters, and output isolation. Application profiles are local and
ignored by Git; copy `configs/apps/app.template.yaml` to create one.

## BBox Annotation Format

Create `data/<app_id>/annotations.csv` with this exact header:

```csv
id,image,description,left,top,right,bottom,app_version
toolbar-acquire-001,main-window.png,The "Acquire" button in the spectrum toolbar,1100,160,1184,210,8.2
```

- `id` is unique within the application and remains stable as data grows.
- `image` is a path relative to the application's `images/` directory.
- `left,top,right,bottom` are integer original-screenshot pixels. Left/top are
  inclusive; right/bottom are exclusive. A valid box satisfies
  `0 <= left < right <= image_width` and `0 <= top < bottom <= image_height`.
- `app_version` is retained in the evaluation report for regression analysis;
  blank or whitespace-only values are normalized to `unknown`.
- The training point is the geometric center of the pixel box:
  `((left + right - 1) / 2, (top + bottom - 1) / 2)`.
- The model response is that center rounded to an integer in the original
  screenshot's pixel space, with a top-left origin. It is not a coordinate in
  the processor's smart-resized image or a fixed output canvas; use the
  original screenshot dimensions when validating or acting on a response.

Screenshots and CSV files are sensitive and ignored by Git. Remove account
names, identifiers, paths, and other retained material before annotation.

## Workflow

1. Run `scripts/check_runtime.sh` and `scripts/copy_model.sh`.
2. Pin the Qwen2.5-VL source with `scripts/pin_sources.sh`.
3. Create a local application profile, then create its image directory and CSV:

   ```bash
   cp configs/apps/app.template.yaml configs/apps/<app_id>.yaml
   ```

   Update `app_id`, `data_root`, and `run_name` in the copied profile, then
   prepare the data:

   ```bash
   python3 scripts/prepare_grounding_data.py --config configs/apps/<app_id>.yaml
   ```

   It accepts any data size of at least two labels, deterministically assigns
   80%/20% train/validation label rows, and records input hashes and split IDs
   in `processed/manifest.json`. It warns when the same screenshot appears in
   both splits; this is expected with row-level splitting but can inflate
   validation results.

4. Create the isolated environment with `scripts/create_environment.sh`, then
   train one application adapter on GPU 0:

   ```bash
   scripts/run_train.sh configs/apps/<app_id>.yaml
   ```

   Change `run_name` in the application YAML before a distinct experiment. The
   resulting `outputs/<app_id>/<run_name>/` keeps deployable adapters separate
   from resumable checkpoints and training records. `adapters/last` is refreshed
   at every checkpoint, while `adapters/best` is refreshed whenever `eval_loss`
   improves. Each adapter has application metadata and can be supplied directly
   to the service or evaluator.

   A run with existing artifacts is never overwritten. To continue an interrupted
   run created with this layout, pass `--resume`; it validates the frozen
   configuration and dataset manifest, then resumes the newest checkpoint:

   ```bash
   scripts/run_train.sh --resume configs/apps/<app_id>.yaml
   ```

   `records/metrics.jsonl` retains step and evaluation metrics independently of
   checkpoint retention. `records/train_<UTC>_<pid>.log` contains the complete
   launcher output, and `records/tensorboard/` contains TensorBoard events.
   Run `scripts/create_environment.sh` once after adopting this layout to add
   the TensorBoard dependency. Older flat-layout runs are intentionally not
   resumable through this command.

5. Evaluate the held-out validation set automatically. The evaluation command
   starts an evaluation-only vLLM service on GPU 1 and port 18001, reads
   `processed/validation.jsonl`, sends each screenshot and grounding prompt to
   the native UI-TARS model, scores the responses, and stops the service when
   it is done:

   ```bash
   /root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora/bin/python \
     scripts/evaluate_grounding.py --config configs/apps/<app_id>.yaml
   ```

   To evaluate an application adapter instead, pass its absolute path. The
   adapter metadata must belong to the selected application:

   ```bash
   /root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora/bin/python \
     scripts/evaluate_grounding.py \
     --config configs/apps/<app_id>.yaml \
     --adapter "$(pwd)/outputs/<app_id>/<run_name>/adapters/best"
   ```

   The automatic mode rejects port 18000, which remains reserved for the
   production service. Use `--port`, `--startup-timeout`, `--request-timeout`,
   or `--max-tokens` only when the defaults need adjustment.

6. The report is written by default to
   `outputs/<app_id>/<run_name>/eval_MMDD_HHMMSS.json`, using UTC in the file
   name. It includes the model mode, total-denominator bbox accuracy,
   parseability, out-of-contract responses, center-distance diagnostics,
   per-example model responses, and grouped metrics by application version.
   `--report` overrides the output path.

   For an existing response file, offline scoring remains available:

   ```bash
   python3 scripts/evaluate_grounding.py \
     --config configs/apps/<app_id>.yaml \
     --responses outputs/<app_id>/<run_name>/responses.jsonl
   ```

## Validation

Validate a profile without loading model weights or data:

```bash
/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora/bin/python \
  scripts/train_grounding.py --config configs/apps/<app_id>.yaml --print-config
```

Run the repository tests with `python3 -m unittest discover -s tests`.
