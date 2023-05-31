#!/usr/bin/env bash
set -e

UI_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null && pwd )"
SNAPSHOT_TOOL=tests/ui_snapshot
SNAPSHOT_DIR=$UI_DIR/tests/snapshots

echo "SNAPSHOT_DIR: $SNAPSHOT_DIR"
mkdir -p $SNAPSHOT_DIR

cd $UI_DIR

echo -n 0 > ~/.comma/params/d/PrimeType
$SNAPSHOT_TOOL $SNAPSHOT_DIR/no-prime.png

echo -n 1 > ~/.comma/params/d/PrimeType
$SNAPSHOT_TOOL $SNAPSHOT_DIR/prime.png
