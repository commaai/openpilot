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
rm -rf /data/safe_staging/* || true

export KEYS_PARAM_PATH="/data/params/d/GithubSshKeys"
export KEYS_PATH="/usr/comma/setup_keys"
export CONTINUE_PATH="/data/continue.sh"

if ! grep -F "$KEYS_PATH" /etc/ssh/sshd_config; then
  echo "setting up keys"
  sudo mount -o rw,remount /
  sudo systemctl enable ssh
  sudo sed -i "s,$KEYS_PARAM_PATH,$KEYS_PATH," /etc/ssh/sshd_config
  sudo mount -o ro,remount /
fi

tee $CONTINUE_PATH << EOF
#!/usr/bin/bash

sudo abctl --set_success

while true; do
  if ! sudo systemctl is-active -q ssh; then
    sudo systemctl start ssh
  fi
  sleep 10s
done

sleep infinity
EOF
chmod +x $CONTINUE_PATH

# set up environment
if [ ! -d "$SOURCE_DIR" ]; then
  git clone https://github.com/commaai/openpilot.git $SOURCE_DIR
fi
cd $SOURCE_DIR

rm -f .git/index.lock
git reset --hard
git fetch --verbose origin $GIT_COMMIT
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;
git reset --hard $GIT_COMMIT
git checkout $GIT_COMMIT
git clean -xdf
git submodule update --init --recursive
git submodule foreach --recursive "git reset --hard && git clean -xdf"

git lfs pull
(ulimit -n 65535 && git lfs prune)

echo "git checkout done, t=$SECONDS"

rsync -a --delete $SOURCE_DIR $TEST_DIR

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
