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

  #if ! pgrep -f 'ciui.py' > /dev/null 2>&1; then
  #  echo 'starting UI'
  #  cp $SOURCE_DIR/selfdrive/test/ciui.py /data/
  #  /data/ciui.py &
  #fi

  sleep 5s
done

sleep infinity
EOF
chmod +x $CONTINUE_PATH

fetch_commit() {
  if git cat-file -e "$GIT_COMMIT^{commit}" 2>/dev/null; then
    echo "$GIT_COMMIT already present"
    return
  fi

  git fetch --no-tags --no-recurse-submodules -j8 --verbose --depth 1 origin "$GIT_COMMIT"
}

submodule_paths() {
  git config --file .gitmodules --get-regexp path 2>/dev/null | awk '{print $2}'
}

submodules_need_update() {
  if [ -z "$OLD_HEAD" ]; then
    return 0
  fi

  local paths
  paths="$(submodule_paths)"

  for path in $paths; do
    if [ ! -e "$path/.git" ]; then
      return 0
    fi
  done

  ! git diff --quiet "$OLD_HEAD" "$GIT_COMMIT" -- .gitmodules $paths
}

lfs_needs_pull() {
  if git lfs ls-files | awk '$2 == "-" { found=1 } END { exit found ? 0 : 1 }'; then
    return 0
  fi

  if [ -z "$OLD_HEAD" ]; then
    return 0
  fi

  if ! git diff --quiet "$OLD_HEAD" "$GIT_COMMIT" -- .gitattributes .lfsconfig; then
    return 0
  fi

  git diff --name-only "$OLD_HEAD" "$GIT_COMMIT" -- | git check-attr --stdin filter | grep -q ': filter: lfs'
}

checkout_common() {
  local clean_args="$1"
  local submodule_clean_args="$2"

  OLD_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"

  # cleanup orphaned locks
  find .git -type f -name "*.lock" -exec rm {} +

  git clean $clean_args
  fetch_commit
  GIT_LFS_SKIP_SMUDGE=1 git -c advice.detachedHead=false checkout --force --detach --no-recurse-submodules "$GIT_COMMIT"
  git clean $clean_args

  if submodules_need_update; then
    git submodule sync --recursive
    git submodule update --init --recursive --force --jobs 8
  else
    echo "submodules unchanged, skipping submodule update"
  fi
  git submodule foreach --recursive "git reset --hard && git clean $submodule_clean_args"

  if lfs_needs_pull; then
    git lfs pull
  else
    echo "LFS files unchanged, skipping git lfs pull"
  fi
}

safe_checkout() {
  # completely clean TEST_DIR

  cd $SOURCE_DIR

  checkout_common "-xdff" "-xdff"

  echo "git checkout done, t=$SECONDS"
  du -hs $SOURCE_DIR $SOURCE_DIR/.git

  rsync -a --delete $SOURCE_DIR $TEST_DIR
}

unsafe_checkout() {( set -e
  # checkout directly in test dir, leave old build products

  cd $TEST_DIR

  checkout_common "-dff" "-df"
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
