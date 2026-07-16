#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora"
PORT="${PORT:-18001}"
APP_ID=""
ADAPTER_PATH=""

usage() {
  echo "usage: $0 --app <app-id> --adapter /absolute/path/to/adapter" >&2
  exit 2
}

while (( $# > 0 )); do
  case "$1" in
    --app)
      (( $# >= 2 )) || usage
      APP_ID="$2"
      shift 2
      ;;
    --adapter)
      (( $# >= 2 )) || usage
      ADAPTER_PATH="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

[[ "$APP_ID" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || { echo "Invalid app ID: $APP_ID" >&2; exit 2; }
[[ -n "$ADAPTER_PATH" && "$ADAPTER_PATH" = /* ]] || { echo "Adapter path must be absolute." >&2; exit 2; }
[[ "$PORT" != "18000" ]] || { echo "Port 18000 is reserved for the production service." >&2; exit 2; }
[[ -x "$CONDA_ENV/bin/vllm" ]] || { echo "Missing isolated environment: $CONDA_ENV" >&2; exit 1; }
[[ -f "$ADAPTER_PATH/adapter_config.json" ]] || { echo "Not a PEFT adapter directory: $ADAPTER_PATH" >&2; exit 1; }
[[ -f "$ADAPTER_PATH/run_metadata.json" ]] || { echo "Missing application metadata: $ADAPTER_PATH/run_metadata.json" >&2; exit 1; }
python3 - "$ADAPTER_PATH/run_metadata.json" "$APP_ID" <<'PY'
import json
import sys

metadata_path, expected_app = sys.argv[1:]
with open(metadata_path, encoding="utf-8") as handle:
    metadata = json.load(handle)
if metadata.get("app_id") != expected_app:
    raise SystemExit(f"Adapter belongs to {metadata.get('app_id')!r}, not {expected_app!r}")
PY
if ss -ltn "sport = :$PORT" | awk 'NR>1 {found=1} END {exit found ? 0 : 1}'; then
  echo "Port $PORT is already in use." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
exec "$CONDA_ENV/bin/vllm" serve "$PROJECT_ROOT/models/UI-TARS-1.5-7B" \
  --served-model-name "ui-tars-1.5-${APP_ID}-lora" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --lora-modules "${APP_ID}-grounding=$ADAPTER_PATH"
