#!/usr/bin/env bash
# Sync openpilot to comma device over SSH
# Usage: ./sync.sh [device-host]
DEVICE=$1

rsync -avz \
  --filter=':- .gitignore' \
  --exclude='.git' \
  ./ $DEVICE:/data/openpilot/

echo "Synced. Restarting openpilot..."
ssh $DEVICE "tmux kill-server 2>/dev/null; pkill -f system.updated.updated; sleep 2; rm -f /tmp/safe_staging_overlay.lock; true"
ssh -t $DEVICE "bash -l -c 'cd /data/openpilot && exec ./launch_openpilot.sh'"