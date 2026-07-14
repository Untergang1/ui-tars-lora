#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_DIR="$PROJECT_ROOT/third_party/Qwen2.5-VL"
AGENT_S_DIR="$PROJECT_ROOT/third_party/Agent-S"

for directory in "$QWEN_DIR" "$AGENT_S_DIR"; do
  if ! git -C "$directory" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "Source clone is incomplete: $directory" >&2
    exit 1
  fi
done

qwen_commit=$(git -C "$QWEN_DIR" rev-parse HEAD)
agent_s_commit=$(git -C "$AGENT_S_DIR" rev-parse HEAD)
cat > "$PROJECT_ROOT/SOURCES.md" <<EOF
# Pinned Upstream Sources

| Source | Remote | Commit | Purpose |
| --- | --- | --- | --- |
| Qwen2.5-VL | https://github.com/QwenLM/Qwen2.5-VL.git | $qwen_commit | Official training entrypoint and dependencies |
| Agent-S | https://github.com/simular-ai/Agent-S.git | $agent_s_commit | UI-TARS grounding prompt and response contract |
EOF

echo "Pinned Qwen2.5-VL: $qwen_commit"
echo "Pinned Agent-S: $agent_s_commit"
