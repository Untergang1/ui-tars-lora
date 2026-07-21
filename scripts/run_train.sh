#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora"
CONFIG_PATH="$PROJECT_ROOT/configs/apps/avantage.yaml"
RESUME_ARGS=()
GPU=0

usage() {
  echo "usage: $0 [--resume] [--gpu <index>] [training-config.yaml]" >&2
  exit 2
}

while (( $# > 0 )); do
  case "$1" in
    --resume)
      RESUME_ARGS=(--resume)
      shift
      ;;
    --gpu)
      (( $# >= 2 )) || usage
      GPU="$2"
      shift 2
      ;;
    --*)
      usage
      ;;
    *)
      [[ "$CONFIG_PATH" == "$PROJECT_ROOT/configs/apps/avantage.yaml" ]] || usage
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU index must be a non-negative integer: $GPU" >&2; exit 2; }
"$PROJECT_ROOT/scripts/check_runtime.sh"
[[ -x "$CONDA_ENV/bin/python" ]] || { echo "Run scripts/create_environment.sh first." >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Training config does not exist: $CONFIG_PATH" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
RECORDS_PATH="$(PYTHONPATH="$PROJECT_ROOT/scripts" "$CONDA_ENV/bin/python" -c \
  'import sys; from pathlib import Path; from training_config import load_training_config; print(load_training_config(Path(sys.argv[1])).records)' \
  "$CONFIG_PATH")"
mkdir -p "$RECORDS_PATH"
LOG_PATH="$RECORDS_PATH/train_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"

set +e
"$CONDA_ENV/bin/python" -u "$PROJECT_ROOT/scripts/train_grounding.py" --config "$CONFIG_PATH" "${RESUME_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e
(( PIPE_RESULTS[1] == 0 )) || exit "${PIPE_RESULTS[1]}"
exit "${PIPE_RESULTS[0]}"
