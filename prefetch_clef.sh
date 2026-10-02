#!/usr/bin/env bash
# Download the Clef-Flash weights into the project's Hugging Face cache.
# Run this once before the first start so startup is fast and download errors are visible.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
ENV_FILE=${ENV_FILE:-"$SCRIPT_DIR/clef.env"}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

MODEL_ID=${MODEL_ID:-Cloudflare/clef-flash}
HF_HOME_HOST=${HF_HOME_HOST:-./data/huggingface}

mkdir -p "$HF_HOME_HOST"

COMPOSE=(docker compose --env-file "$ENV_FILE" -f "$SCRIPT_DIR/compose.yaml")

echo "Building image if needed..."
"${COMPOSE[@]}" build

echo "Prefetching $MODEL_ID into $HF_HOME_HOST ..."
"${COMPOSE[@]}" run --rm --no-deps --entrypoint python clef-flash -c \
  'import os; from huggingface_hub import snapshot_download; print("downloaded to", snapshot_download(os.environ["MODEL_ID"]))'

echo "Prefetch complete. Disk usage:"
du -sh "$HF_HOME_HOST"
