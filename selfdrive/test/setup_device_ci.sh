#!/usr/bin/bash -e

export SOURCE_DIR="/data/openpilot_source/"

if [ -z "$GIT_COMMIT" ]; then
  echo "GIT_COMMIT must be set"
  exit 1
fi

if [ -z "$TEST_DIR" ]; then
  echo "TEST_DIR must be set"
  exit 1
fi

if [ -f "/EON" ]; then
  rm -rf /data/core
  rm -rf /data/neoupdate
  rm -rf /data/safe_staging
fi

# set up environment
cd $SOURCE_DIR
git fetch origin $GIT_COMMIT
git reset --hard $GIT_COMMIT
git checkout $GIT_COMMIT
git clean -xdf
git submodule update --init --recursive
git submodule foreach --recursive "git reset --hard && git clean -xdf"

echo "git checkout done, t=$SECONDS"

rsync -a --delete $SOURCE_DIR $TEST_DIR

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
