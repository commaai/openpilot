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

# prevent storage from filling up
rm -rf /data/media/0/realdata/*

rm -rf /data/safe_staging/ || true
if [ -d /data/safe_staging/ ]; then
  sudo umount /data/safe_staging/merged/ || true
  rm -rf /data/safe_staging/ || true
fi

CONTINUE_PATH="/data/continue.sh"
tee $CONTINUE_PATH << EOF
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
EOF
chmod +x $CONTINUE_PATH

export GIT_PACK_THREADS=8
# write LFS pointers during checkout, then batch download in git lfs pull
export GIT_LFS_SKIP_SMUDGE=1

# NOTE: bash ignores set -e inside functions invoked on the left of ||, even when
# re-asserted, so the bodies are explicit && chains ending in a HEAD assertion.

# bring the repo in $1 to $GIT_COMMIT; $2 is the set of git-clean flags
sync_repo() {( set -e
  cd "$1" &&
  CLEAN_FLAGS="$2" &&

  # cleanup orphaned locks
  find .git -type f -name "*.lock" -exec rm {} + &&

  git fetch --no-tags --no-recurse-submodules --depth 1 origin $GIT_COMMIT &&
  git -c checkout.workers=8 checkout --force --no-recurse-submodules $GIT_COMMIT &&
  # release builds and sudo'd tests leave root-owned files behind
  { git clean $CLEAN_FLAGS || { sudo chown -R comma: . && git clean $CLEAN_FLAGS; }; } &&
  git submodule sync &&
  git -c checkout.workers=8 submodule update --init --recursive --force --jobs 6 &&
  git submodule foreach --recursive "git reset --hard && git clean $CLEAN_FLAGS" &&
  git lfs pull &&
  (ulimit -n 65535 && git lfs prune) &&

  [ "$(git rev-parse HEAD)" = "$GIT_COMMIT" ]
)}

safe_checkout() {( set -e
  # completely clean TEST_DIR: bring SOURCE_DIR to a pristine $GIT_COMMIT, then mirror it.
  # both the checkout and the mirror are incremental, so this is fast for small diffs.
  sync_repo $SOURCE_DIR "-xdff" &&

  echo "git checkout done, t=$SECONDS" &&

  # pack files are content-addressed (same name = same content), but git freshens their
  # mtimes on object reuse, so sync them by name only or a plain rsync re-copies the big
  # base pack on every run. refs/HEAD sync last, so TEST_DIR is never ahead of its objects.
  mkdir -p $TEST_DIR/.git/objects/pack &&
  rsync -a --delete --ignore-existing --include='pack-*' --exclude='*' ${SOURCE_DIR%/}/.git/objects/pack/ $TEST_DIR/.git/objects/pack/ &&
  rsync -a --delete --exclude='/.git/objects/pack/pack-*' $SOURCE_DIR $TEST_DIR &&

  [ "$(git -C $TEST_DIR rev-parse HEAD)" = "$GIT_COMMIT" ]
)}

unsafe_checkout() {( set -e
  # checkout directly in TEST_DIR, leaving old build products around
  sync_repo $TEST_DIR "-dff"
)}

nuke_checkout() {( set -e
  # last resort: start over from scratch.
  # nuking can't fix a network failure or an unfetchable commit (GitHub only
  # serves advertised tips), so don't throw away good local state for those.
  git ls-remote https://github.com/commaai/openpilot.git | grep -q "^$GIT_COMMIT"

  # the shell may be sitting inside one of the dirs we're about to delete
  cd /

  sudo rm -rf $SOURCE_DIR $TEST_DIR
  git clone --depth 1 https://github.com/commaai/openpilot.git $SOURCE_DIR
  safe_checkout
)}

# set up environment
if [ ! -d "$SOURCE_DIR" ]; then
  git clone --depth 1 https://github.com/commaai/openpilot.git $SOURCE_DIR
fi

if [ ! -z "$UNSAFE" ]; then
  echo "trying unsafe checkout"
  unsafe_checkout || safe_checkout || nuke_checkout
else
  echo "doing safe checkout"
  safe_checkout || nuke_checkout
fi

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
