#!/usr/bin/env bash
# Stop and remove the dockerized Clef-Flash inference API.
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

COMPOSE=(docker compose --env-file "$ENV_FILE" -f "$SCRIPT_DIR/compose.yaml")

"${COMPOSE[@]}" down
echo "Stopped 'clef-flash'. Model/cache data was left intact."
