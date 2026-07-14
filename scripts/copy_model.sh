#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_SNAPSHOT="/root/autodl-tmp/xukefan/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e"
DESTINATION="$PROJECT_ROOT/models/UI-TARS-1.5-7B"

"$PROJECT_ROOT/scripts/check_runtime.sh"

if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION" >&2
  echo "Refusing to overwrite a model copy." >&2
  exit 1
fi

mkdir -p "$DESTINATION"
# Resolve Hugging Face snapshot symlinks and never create hard links to the live cache.
cp --reflink=auto --dereference --preserve=mode,timestamps -R "$SOURCE_SNAPSHOT"/. "$DESTINATION"/

if find "$DESTINATION" -type l -print -quit | rg -q .; then
  echo "Model copy unexpectedly contains symlinks; refusing to mark it ready." >&2
  exit 1
fi

if [[ ! -f "$DESTINATION/config.json" || ! -f "$DESTINATION/model.safetensors.index.json" ]]; then
  echo "Copied model is incomplete." >&2
  exit 1
fi

{
  echo "source_snapshot=$SOURCE_SNAPSHOT"
  echo "copied_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source_config_sha256=$(sha256sum "$SOURCE_SNAPSHOT/config.json" | awk '{print $1}')"
  echo "destination_config_sha256=$(sha256sum "$DESTINATION/config.json" | awk '{print $1}')"
  echo "size=$(du -sh "$DESTINATION" | awk '{print $1}')"
} > "$DESTINATION/COPY_MANIFEST.txt"

# The base model is immutable; adapters are always written under outputs/.
chmod -R a-w "$DESTINATION"

echo "Created independent, read-only UI-TARS model copy: $DESTINATION"
cat "$DESTINATION/COPY_MANIFEST.txt"
