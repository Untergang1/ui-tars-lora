#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_SNAPSHOT="/root/autodl-tmp/xukefan/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e"
SERVICE_PID=841653
MIN_FREE_GIB=70

if [[ ! -f "$SOURCE_SNAPSHOT/config.json" ]]; then
  echo "Missing UI-TARS source snapshot: $SOURCE_SNAPSHOT" >&2
  exit 1
fi

if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
  echo "Expected production vLLM PID $SERVICE_PID is not running; stop before continuing." >&2
  exit 1
fi

if ! ps -p "$SERVICE_PID" -o cmd= | rg -q 'vllm serve ByteDance-Seed/UI-TARS-1.5-7B'; then
  echo "PID $SERVICE_PID is not the expected UI-TARS vLLM service; stop before continuing." >&2
  exit 1
fi

available_gib=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
if (( available_gib < MIN_FREE_GIB )); then
  echo "Only ${available_gib}GiB available; at least ${MIN_FREE_GIB}GiB is required." >&2
  exit 1
fi

echo "UI-TARS source snapshot: $SOURCE_SNAPSHOT"
echo "Production vLLM PID: $SERVICE_PID (GPU 2 must remain reserved)"
echo "Available project storage: ${available_gib}GiB"
echo "GPU state:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
