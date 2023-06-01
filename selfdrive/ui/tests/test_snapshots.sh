#!/usr/bin/env bash
UI_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null && pwd )"
SNAPSHOT_TOOL=tests/ui_snapshot
SNAPSHOT_DIR=$UI_DIR/tests/snapshots

cd $UI_DIR

for TEST_SETUP in $SNAPSHOT_DIR/*.sh; do
  TEST_CASE=$(basename ${TEST_SETUP%.sh})
  TEST_SNAPSHOT=$SNAPSHOT_DIR/$TEST_CASE.png
  TMP_SNAPSHOT=$(mktemp /tmp/snapshot.XXXXXXXXXX).png

  echo "Checking $TEST_CASE..."

  if [ ! -f $TEST_SNAPSHOT ]; then
    >&2 echo "Missing snapshot! Create with: rebase_snapshot.sh $TEST_CASE"
    exit 1
  fi

  bash $SNAPSHOT_DIR/base.sh
  bash $TEST_SETUP
  $SNAPSHOT_TOOL $TMP_SNAPSHOT

  compare -metric AE $TEST_SNAPSHOT $TMP_SNAPSHOT null: 2> /dev/null
  if [ $? -ne 0 ]; then
    >&2 echo "Snapshot changed! Update with: rebase_snapshot.sh $TEST_CASE"
    >&2 echo "Original: $TEST_SNAPSHOT"
    >&2 echo "Output: $TMP_SNAPSHOT"
    exit 1
  fi
done
