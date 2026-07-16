#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora"
CONFIG_PATH="${1:-$PROJECT_ROOT/configs/apps/avantage.yaml}"

if (( $# > 1 )); then
  echo "usage: $0 [training-config.yaml]" >&2
  exit 2
fi

"$PROJECT_ROOT/scripts/check_runtime.sh"
[[ -x "$CONDA_ENV/bin/python" ]] || { echo "Run scripts/create_environment.sh first." >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "Training config does not exist: $CONFIG_PATH" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
exec "$CONDA_ENV/bin/python" "$PROJECT_ROOT/scripts/train_grounding.py" --config "$CONFIG_PATH"
