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

# TODO: never clear qcom_replay cache
# clear scons cache dirs that haven't been written to in one day
cd /tmp && find -name 'scons_cache_*' -type d -maxdepth 1 -mtime +1 -exec rm -rf '{}' \;

# this can get really big on the CI devices
rm -rf /data/core

# set up environment
cd $SOURCE_DIR
git reset --hard
git fetch origin
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;
git reset --hard $GIT_COMMIT
git checkout $GIT_COMMIT
git clean -xdf
git submodule update --init
git submodule foreach --recursive git reset --hard
git submodule foreach --recursive git clean -xdf
echo "git checkout took $SECONDS seconds"

rsync -a --delete $SOURCE_DIR $TEST_DIR

echo "$TEST_DIR synced with $GIT_COMMIT, took $SECONDS seconds"
