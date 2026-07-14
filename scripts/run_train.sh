#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora"

"$PROJECT_ROOT/scripts/check_runtime.sh"
[[ -x "$CONDA_ENV/bin/python" ]] || { echo "Run scripts/create_environment.sh first." >&2; exit 1; }
[[ -f "$PROJECT_ROOT/data/processed/train.jsonl" ]] || { echo "Run scripts/prepare_grounding_data.py first." >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
exec "$CONDA_ENV/bin/python" "$PROJECT_ROOT/scripts/train_grounding.py"
