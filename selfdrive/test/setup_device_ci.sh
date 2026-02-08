#!/usr/bin/env bash

set -e
set -x

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

setup_device() {
  # prevent storage from filling up
  rm -rf /data/media/0/realdata/*

  rm -rf /data/safe_staging/ || true
  if [ -d /data/safe_staging/ ]; then
    sudo umount /data/safe_staging/merged/ || true
    rm -rf /data/safe_staging/ || true
  fi

  CONTINUE_PATH="/data/continue.sh"
  cat > $CONTINUE_PATH << 'CONTINUE_EOF'
#!/usr/bin/env bash

sudo abctl --set_success

# patch sshd config
sudo mount -o rw,remount /
sudo sed -i "s,/data/params/d/GithubSshKeys,/usr/comma/setup_keys," /etc/ssh/sshd_config
sudo systemctl daemon-reload
sudo systemctl restart ssh
sudo systemctl restart NetworkManager
sudo systemctl disable ssh-param-watcher.path
sudo systemctl disable ssh-param-watcher.service
sudo mount -o ro,remount /
sudo systemctl stop power_monitor

while true; do
  if ! sudo systemctl is-active -q ssh; then
    sudo systemctl start ssh
  fi

  sleep 5s
done

sleep infinity
CONTINUE_EOF
  chmod +x $CONTINUE_PATH
}

safe_checkout() {
  # completely clean TEST_DIR

  cd $SOURCE_DIR

  # cleanup orphaned locks
  find .git -type f -name "*.lock" -exec rm {} +

  git fetch --no-tags --no-recurse-submodules -j4 --depth 1 origin $GIT_COMMIT
  git checkout --force $GIT_COMMIT
  git clean -xdff
  git submodule update --init --recursive --force -j$(nproc)
  git submodule foreach --recursive "git clean -xdff"

  git lfs pull

  echo "git checkout done, t=$SECONDS"
  du -hs $SOURCE_DIR $SOURCE_DIR/.git

  rsync -a --delete $SOURCE_DIR $TEST_DIR
}

ms() { echo $(( $(date +%s%N) / 1000000 )); }

unsafe_checkout() {( set -e
  # checkout directly in test dir, leave old build products

  cd $TEST_DIR
  _t0=$(ms)

  # skip everything if already at the target commit
  if [ "$(git rev-parse HEAD 2>/dev/null)" == "$GIT_COMMIT" ]; then
    echo "== already at $GIT_COMMIT, skipping checkout"
    return 0
  fi

  # cleanup orphaned locks
  find .git -type f -name "*.lock" -exec rm {} +

  git fetch --no-tags --no-recurse-submodules -j8 --depth 1 origin $GIT_COMMIT
  echo "== fetch $(( $(ms) - _t0 ))ms"

  git checkout --force --no-recurse-submodules $GIT_COMMIT
  git clean -dff
  echo "== checkout+clean $(( $(ms) - _t0 ))ms"

  # update submodules and pull lfs in parallel
  (
    if git submodule status | grep -q '^[+-]'; then
      git submodule update --init --recursive --force -j$(nproc)
    fi
  ) &
  _sub_pid=$!
  git lfs pull &
  _lfs_pid=$!
  wait $_sub_pid
  echo "== submodule $(( $(ms) - _t0 ))ms"
  wait $_lfs_pid
  echo "== lfs $(( $(ms) - _t0 ))ms"
)}

export GIT_PACK_THREADS=8

# set up environment
if [ ! -d "$SOURCE_DIR" ]; then
  git clone https://github.com/commaai/openpilot.git $SOURCE_DIR
fi

# run device setup in parallel with checkout
setup_device &
_setup_pid=$!

if [ ! -z "$UNSAFE" ]; then
  echo "trying unsafe checkout"
  set +e
  unsafe_checkout
  if [[ "$?" -ne 0 ]]; then
    safe_checkout
  fi
  set -e
else
  echo "doing safe checkout"
  safe_checkout
fi

wait $_setup_pid

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
