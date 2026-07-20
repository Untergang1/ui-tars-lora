#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_DIR="$PROJECT_ROOT/third_party/Qwen2.5-VL"

if ! git -C "$QWEN_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "Source clone is incomplete: $QWEN_DIR" >&2
  exit 1
fi

qwen_commit=$(git -C "$QWEN_DIR" rev-parse HEAD)
cat > "$PROJECT_ROOT/SOURCES.md" <<EOF
# Pinned Upstream Sources

| Source | Remote | Commit | Purpose |
| --- | --- | --- | --- |
| Qwen2.5-VL | https://github.com/QwenLM/Qwen2.5-VL.git | $qwen_commit | Official training entrypoint and dependencies |
EOF

echo "Pinned Qwen2.5-VL: $qwen_commit"
