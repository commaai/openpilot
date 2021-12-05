#!/usr/bin/bash

set -e

if [ -z "$SOURCE_DIR" ]; then
  echo "SOURCE_DIR must be set"
  exit 1
fi

if [ -z "$GIT_COMMIT" ]; then
  echo "GIT_COMMIT must be set"
  exit 1
fi

if [ -z "$TEST_DIR" ]; then
  echo "TEST_DIR must be set"
  exit 1
fi

umount /data/safe_staging/merged/ || true
sudo umount /data/safe_staging/merged/ || true

if [ -f "/EON" ]; then
  rm -rf /data/core
  rm -rf /data/neoupdate
  rm -rf /data/safe_staging
fi

export CONTINUE_PATH="/data/continue.sh"
if [ -f "/EON" ]; then
  export CONTINUE_PATH="/data/data/com.termux/files/continue.sh"
fi
tee $CONTINUE_PATH << ABC
#!/usr/bin/bash

PARAMS_ROOT="/data/params/d"

while true; do
  mkdir -p $PARAMS_ROOT
  cp /data/data/com.termux/files/home/setup_keys $PARAMS_ROOT/GithubSshKeys
  echo -n 1 > $PARAMS_ROOT/SshEnabled
  sleep 1m
done

sleep infinity
ABC

# set up environment
if [ ! -d "$SOURCE_DIR" ]; then
  git clone https://github.com/commaai/openpilot.git $SOURCE_DIR
fi
cd $SOURCE_DIR
git reset --hard
git fetch
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;
git fetch --verbose origin $GIT_COMMIT
git reset --hard $GIT_COMMIT
git checkout $GIT_COMMIT
git clean -xdf
git submodule update --init --recursive
git submodule foreach --recursive "git reset --hard && git clean -xdf"

echo "git checkout done, t=$SECONDS"

rsync -a --delete $SOURCE_DIR $TEST_DIR

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
