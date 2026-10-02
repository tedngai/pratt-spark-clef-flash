#!/usr/bin/env bash
# Install and enable the Clef-Flash systemd unit for the current project directory.
# Usage: sudo ./install_systemd.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-"$SCRIPT_DIR"}
SERVICE_NAME=${SERVICE_NAME:-clef-flash}
DOCKER_BIN=${DOCKER_BIN:-$(command -v docker || true)}

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (use sudo)." >&2
  exit 1
fi
if [[ -z "$DOCKER_BIN" ]]; then
  echo "docker not found in PATH." >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/compose.yaml" ]]; then
  echo "compose.yaml not found in $PROJECT_DIR" >&2
  exit 1
fi

TEMPLATE="$SCRIPT_DIR/systemd/clef-flash.service"
UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__DOCKER_BIN__|$DOCKER_BIN|g" \
    "$TEMPLATE" > "$UNIT"

systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

echo "Installed $UNIT"
echo "Status: systemctl status $SERVICE_NAME"
echo "Logs:   journalctl -u $SERVICE_NAME -f"
