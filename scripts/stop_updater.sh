#!/usr/bin/env sh

# Stop updater
pkill -2 -f selfdrive.updated.updated

# Remove pending update
rm -f /data/safe_staging/finalized/.overlay_consistent
