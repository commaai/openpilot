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

CONTINUE_PATH="/data/continue.sh"
tee $CONTINUE_PATH << EOF
#!/usr/bin/bash

sudo abctl --set_success

# patch sshd config
sudo mount -o rw,remount /
echo tici-$(cat /proc/cmdline | sed -e 's/^.*androidboot.serialno=//' -e 's/ .*$//') | sudo tee /etc/hostname
sudo sed -i "s,/data/params/d/GithubSshKeys,/usr/comma/setup_keys," /etc/ssh/sshd_config
sudo systemctl daemon-reload
sudo systemctl restart ssh
sudo systemctl disable ssh-param-watcher.path
sudo systemctl disable ssh-param-watcher.service
sudo mount -o ro,remount /

while true; do
  if ! sudo systemctl is-active -q ssh; then
    sudo systemctl start ssh
  fi
  sleep 5s
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
git fetch --no-tags --no-recurse-submodules -j4 --verbose --depth 1 origin $GIT_COMMIT
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;
git reset --hard $GIT_COMMIT
git checkout $GIT_COMMIT
git clean -xdff
git submodule sync
git submodule update --init --recursive
git submodule foreach --recursive "git reset --hard && git clean -xdff"

git lfs pull
(ulimit -n 65535 && git lfs prune)

echo "git checkout done, t=$SECONDS"
du -hs $SOURCE_DIR $SOURCE_DIR/.git

rsync -a --delete $SOURCE_DIR $TEST_DIR

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
