#!/usr/bin/env bash
# Disable and remove the Clef-Flash systemd unit.
# Usage: sudo ./uninstall_systemd.sh
set -euo pipefail

SERVICE_NAME=${SERVICE_NAME:-clef-flash}
UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (use sudo)." >&2
  exit 1
fi

systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
rm -f "$UNIT"
systemctl daemon-reload
echo "Removed $UNIT"
