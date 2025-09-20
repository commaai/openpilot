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

while true; do
  if ! sudo systemctl is-active -q ssh; then
    sudo systemctl start ssh
  fi

  #if ! pgrep -f 'ciui.py' > /dev/null 2>&1; then
  #  echo 'starting UI'
  #  cp $SOURCE_DIR/selfdrive/test/ciui.py /data/
  #  /data/ciui.py &
  #fi

  awk '{print \$1}' /proc/uptime > /var/tmp/power_watchdog
  sleep 5s
done

sleep infinity
EOF
chmod +x $CONTINUE_PATH

safe_checkout() {
  # completely clean TEST_DIR

  cd $SOURCE_DIR

  # cleanup orphaned locks
  find .git -type f -name "*.lock" -exec rm {} +

  git reset --hard
  git fetch --no-tags --no-recurse-submodules -j4 --verbose --depth 1 origin $GIT_COMMIT
  find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;
  git reset --hard $GIT_COMMIT
  git checkout $GIT_COMMIT
  git clean -xdff
  git submodule sync
  git submodule foreach --recursive "git reset --hard && git clean -xdff"
  git submodule update --init --recursive
  git submodule foreach --recursive "git reset --hard && git clean -xdff"

  git lfs pull
  (ulimit -n 65535 && git lfs prune)

  echo "git checkout done, t=$SECONDS"
  du -hs $SOURCE_DIR $SOURCE_DIR/.git

  rsync -a --delete $SOURCE_DIR $TEST_DIR
}

unsafe_checkout() {( set -e
  # checkout directly in test dir, leave old build products

  cd $TEST_DIR

  # cleanup orphaned locks
  find .git -type f -name "*.lock" -exec rm {} +

  git fetch --no-tags --no-recurse-submodules -j8 --verbose --depth 1 origin $GIT_COMMIT
  git checkout --force --no-recurse-submodules $GIT_COMMIT
  git reset --hard $GIT_COMMIT
  git clean -dff
  git submodule sync
  git submodule foreach --recursive "git reset --hard && git clean -df"
  git submodule update --init --recursive
  git submodule foreach --recursive "git reset --hard && git clean -df"

  git lfs pull
  (ulimit -n 65535 && git lfs prune)
)}

export GIT_PACK_THREADS=8

# set up environment
if [ ! -d "$SOURCE_DIR" ]; then
  git clone https://github.com/commaai/openpilot.git $SOURCE_DIR
fi

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

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
