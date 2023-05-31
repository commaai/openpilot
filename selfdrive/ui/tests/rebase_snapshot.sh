#!/usr/bin/env bash
set -e

UI_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null && pwd )"
SNAPSHOT_TOOL=tests/ui_snapshot
SNAPSHOT_DIR=$UI_DIR/tests/snapshots

cd $UI_DIR

TEST_CASE=$1

if [ -z "$TEST_CASE" ]; then
  >&2 echo "Usage: rebase_snapshot.sh <test_case>"
  exit 1
fi

TEST_SETUP=$SNAPSHOT_DIR/$TEST_CASE.sh
TEST_SNAPSHOT=$SNAPSHOT_DIR/$TEST_CASE.png

echo "Updating $TEST_CASE..."

bash $SNAPSHOT_DIR/base.sh
bash $TEST_SETUP
$SNAPSHOT_TOOL $TEST_SNAPSHOT
