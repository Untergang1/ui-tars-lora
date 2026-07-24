# UI-TARS Per-Application Grounding LoRA

This project fine-tunes the local `ByteDance-Seed/UI-TARS-1.5-7B` model for
UI grounding while keeping one LoRA adapter per application:

```text
screenshot + UI element description -> single-coordinate response
```

The grounding contract uses a single coordinate response. Labels include a full
bounding box: its geometric center supplies the single-point training target,
and validation considers a prediction correct when it falls inside the box. The
production vLLM service remains on GPU 2 and port 18000. Training defaults to
GPU 0, while temporary adapter evaluation and the standalone adapter service
default to GPU 1; each entrypoint accepts `--gpu <index>` for manual placement.

## Application Layout

Every application has a separate ignored data root:

```text
data/<app_id>/
  v1/                     # complete, immutable dataset snapshot
    images/               # sensitive PNG/JPEG screenshots
    annotations.csv       # sensitive bbox labels
    processed/            # generated JSONL and manifest
  v2/                     # next complete snapshot
    images/
    annotations.csv
    processed/
outputs/<app_id>/<run_name>/
  adapters/
    last/               # latest deployable PEFT adapter
    best/               # lowest validation-loss deployable adapter
  checkpoints/          # resumable Trainer state, retained per save_total_limit
  records/              # frozen config/data, JSONL metrics, text logs, TensorBoard events
    eval/               # one lightweight generated-coordinate report per epoch
configs/apps/<app_id>.yaml
```

`<app_id>` is a stable lowercase slug such as `avantage` or `omnic`.
`dataset_version` selects a complete snapshot named `v1`, `v2`, `v3`, and so
on. The application YAML is the single source of truth for data paths, split
settings, training parameters, and output isolation. Application profiles are
local and ignored by Git; copy `configs/apps/app.template.yaml` to create one.

## BBox Annotation Format

Create `data/<app_id>/<dataset_version>/annotations.csv` with this exact header:

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

## Dataset Versions

Each `vN` directory is a complete snapshot. Never edit an existing version;
create the next version by copying its predecessor, then change only the new
copy and regenerate its `processed/` directory:

```bash
cp -a data/<app_id>/v1 data/<app_id>/v2
# Update data/<app_id>/v2/images/ and data/<app_id>/v2/annotations.csv.
# Set dataset_version: v2 and a new run_name in the local application YAML.
python3 scripts/prepare_grounding_data.py --config configs/apps/<app_id>.yaml
```

For a legacy flat application root, first move `images/`, `annotations.csv`,
and `processed/` into `v1/`, then regenerate `v1/processed/`. This rebuild is
required because generated records contain resolved image paths. Existing output
runs remain historical artifacts and must not be resumed with the migrated
configuration.

## Workflow

1. Run `scripts/check_runtime.sh` and `scripts/copy_model.sh`.
2. Pin the Qwen2.5-VL source with `scripts/pin_sources.sh`.
3. Create a local application profile, then create its image directory and CSV:

   ```bash
   cp configs/apps/app.template.yaml configs/apps/<app_id>.yaml
   ```

   Update `app_id`, `data_root`, `dataset_version`, and `run_name` in the copied
   profile, then prepare the selected snapshot:

   ```bash
   python3 scripts/prepare_grounding_data.py --config configs/apps/<app_id>.yaml
   ```

   Set `vision_projector_lora: true` in the profile to additionally train LoRA
   adapters on UI-TARS's visual merger MLP. It remains `false` by default, so
   existing profiles continue to train only the language-model targets; the
   vision encoder blocks stay excluded in either mode.

   It accepts any data size of at least two labels, deterministically assigns
   80%/20% train/validation label rows, and records input hashes and split IDs
   in `<dataset_version>/processed/manifest.json`. The manifest is bound to its
   application and dataset version, so training and evaluation reject a
   different snapshot. It warns when the same screenshot appears in both splits;
   this is expected with row-level splitting but can inflate validation results.

4. Create the isolated environment with `scripts/create_environment.sh`, then
   train one application adapter (GPU 0 by default):

   ```bash
   scripts/run_train.sh --gpu 0 configs/apps/<app_id>.yaml
   ```

   Change `run_name` before a distinct experiment, including when changing
   dataset versions (for example, `data-v1` and `data-v2`). The resulting
   `outputs/<app_id>/<run_name>/` keeps deployable adapters separate from
   resumable checkpoints and training records. `adapters/last` is refreshed at
   every checkpoint, while `adapters/best` is refreshed whenever `eval_loss`
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
   After each completed epoch eval, training also writes
   `records/eval/epoch_<epoch>.json`. This compact JSON contains that epoch's
   token-level `eval_loss`, aggregate coordinate quality metrics, and one
   record per validation label with only `id`, `bbox_hit`,
   `pixel_distance_to_bbox_center`, and `relative_diagonal_error`. The two
   distances are `null` for unparseable or out-of-range coordinates; no model
   responses, prompts, screenshots, or bbox labels are copied into this file.
   Re-running an epoch while resuming training atomically replaces its report.
   Run `scripts/create_environment.sh` once after adopting this layout to add
   the TensorBoard dependency. Older flat-layout runs are intentionally not
   resumable through this command.

5. Evaluate the held-out validation set automatically. The command reads
   `processed/validation.jsonl`, sends each screenshot and grounding prompt to
   the current run's `adapters/best` LoRA adapter, and scores the responses.
   By default, language-only adapters run through an evaluation-only vLLM service
   on GPU 1 and port 18001. Adapters that target `visual.*` modules run through
   local Transformers/PEFT inference so vLLM cannot silently ignore their visual
   LoRA weights:

   ```bash
   /root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora/bin/python \
     scripts/evaluate_grounding.py --config configs/apps/<app_id>.yaml --gpu 1
   ```

   Use `--backend vllm` or `--backend transformers` to override automatic
   selection. vLLM rejects adapters with visual LoRA rather than evaluating them
   with ignored weights. The Transformers backend does not start a service, so
   `--port`, `--startup-timeout`, and `--request-timeout` apply only to vLLM.

   If `outputs/<app_id>/<run_name>/adapters/` does not exist, the same command
   evaluates the native UI-TARS model and records `adapter: null` plus the
   fallback reason in the report. If that directory exists, it must contain a
   valid `best` adapter; evaluation fails rather than silently falling back.

   To override the adapter selected from the configuration, pass its path.
   Relative paths are resolved from the current working directory before
   inference starts, and the adapter metadata must belong to the selected application:

   ```bash
   /root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora/bin/python \
     scripts/evaluate_grounding.py \
     --config configs/apps/<app_id>.yaml \
     --adapter outputs/<app_id>/<run_name>/adapters/best
   ```

   To expose a language-only adapter as a persistent local vLLM service instead, use:

   ```bash
   scripts/serve_adapter.sh --app <app_id> \
     --adapter "$(pwd)/outputs/<app_id>/<run_name>/adapters/best" --gpu 1
   ```

   vLLM cannot serve visual LoRA targets, so use the Transformers evaluation
   backend for adapters created with `vision_projector_lora: true`.

   The vLLM backend rejects port 18000, which remains reserved for the
   production service. Use `--gpu` to select any GPU index (including GPU 2)
   when manually scheduling the workload. Use `--port`, `--startup-timeout`,
   `--request-timeout`, or `--max-tokens` only when the defaults need adjustment.

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
