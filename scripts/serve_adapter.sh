#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="/root/autodl-tmp/xukefan/miniconda3/envs/ui-tars-lora"
ADAPTER_PATH="${1:?usage: $0 /absolute/path/to/adapter}"
PORT="${PORT:-18001}"

if [[ ! -x "$CONDA_ENV/bin/vllm" ]]; then
  echo "Missing isolated environment: $CONDA_ENV" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
  echo "Not a PEFT adapter directory: $ADAPTER_PATH" >&2
  exit 1
fi
if ss -ltn "sport = :$PORT" | awk 'NR>1 {found=1} END {exit found ? 0 : 1}'; then
  echo "Port $PORT is already in use." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
exec "$CONDA_ENV/bin/vllm" serve "$PROJECT_ROOT/models/UI-TARS-1.5-7B" \
  --served-model-name ui-tars-1.5-avantage-lora \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --lora-modules avantage-grounding="$ADAPTER_PATH"
