#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="/root/autodl-tmp/xukefan/miniconda3"
SOURCE_PREFIX="$CONDA_ROOT/envs/ui-tars-vllm"
TARGET_PREFIX="$CONDA_ROOT/envs/ui-tars-lora"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

if [[ -d "$TARGET_PREFIX" ]]; then
  echo "Environment already exists: $TARGET_PREFIX"
else
  # A filesystem copy avoids Anaconda-channel Terms-of-Service prompts and
  # never changes the environment that runs the production vLLM service.
  cp --reflink=auto -a "$SOURCE_PREFIX" "$TARGET_PREFIX"
fi

"$TARGET_PREFIX/bin/python" -m pip install --index-url "$PYPI_MIRROR" \
  'peft==0.14.0' \
  'accelerate==1.3.0' \
  'datasets==3.2.0' \
  'bitsandbytes==0.45.0' \
  'qwen-vl-utils==0.0.10' \
  'PyYAML==6.0.3'

"$TARGET_PREFIX/bin/python" - <<'PY'
import importlib.metadata as metadata
for package in ('torch', 'transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes', 'qwen-vl-utils', 'PyYAML', 'xformers'):
    print(f'{package}={metadata.version(package)}')
PY
mkdir -p "$PROJECT_ROOT/.cache/huggingface"
