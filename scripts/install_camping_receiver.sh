#!/bin/bash
set -euo pipefail

DEST=/data/camping/bin
echo "Installing receiver binaries to $DEST"
mkdir -p "$DEST"

copy_if_exists() {
  local src="$1"; local dst_name="$2"
  if [ -x "$src" ]; then
    install -m 0755 "$src" "$DEST/$dst_name"
    echo "installed $(basename "$src") -> $DEST/$dst_name"
  fi
}

# Open Screen cast receiver
copy_if_exists selfdrive/camping/bin/openscreen-cast-receiver openscreen-cast-receiver
# DLNA renderer
copy_if_exists selfdrive/camping/bin/gmediarender gmediarender
# MiracleCast daemons/tools
copy_if_exists selfdrive/camping/bin/miracle-wifid miracle-wifid
copy_if_exists selfdrive/camping/bin/miracle-wfdctl miracle-wfdctl
copy_if_exists selfdrive/camping/bin/miracle-sinkctl miracle-sinkctl

echo "Done. Ensure files exist and are executable on device."
