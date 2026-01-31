#!/usr/bin/env bash
# Sync openpilot to comma device over SSH
# Usage: ./sync.sh [device-host]
DEVICE=${1:-comma}

rsync -avz \
  --filter=':- .gitignore' \
  --exclude='.git' \
  ./ $DEVICE:/data/openpilot/

echo "Synced. Restarting openpilot..."
ssh $DEVICE "pkill -f launch_chffrplus; pkill -f manager.py; sleep 2; tmux kill-server 2>/dev/null; true"
ssh -t $DEVICE "bash -l -c 'cd /data/openpilot && exec ./launch_openpilot.sh'"