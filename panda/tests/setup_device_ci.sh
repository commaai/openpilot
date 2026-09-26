#!/usr/bin/env bash

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

CONTINUE_PATH="/data/continue.sh"
tee $CONTINUE_PATH << EOF
#!/usr/bin/env bash

sudo abctl --set_success

# patch sshd config
sudo mount -o rw,remount /
sudo sed -i "s,/data/params/d/GithubSshKeys,/usr/comma/setup_keys," /etc/ssh/sshd_config
sudo systemctl daemon-reload
sudo systemctl restart ssh
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

if [ ! -d "$SOURCE_DIR" ]; then
  git clone https://github.com/commaai/panda.git $SOURCE_DIR
fi

# setup device/SOM state
SOM_ST_IO=49
echo $SOM_ST_IO > /sys/class/gpio/export || true
echo out > /sys/class/gpio/gpio${SOM_ST_IO}/direction
echo 1 > /sys/class/gpio/gpio${SOM_ST_IO}/value

# checkout panda commit
cd $SOURCE_DIR

rm -f .git/index.lock
git reset --hard
git fetch --no-tags --no-recurse-submodules -j4 --verbose --depth 1 origin $GIT_COMMIT
find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \;
git reset --hard $GIT_COMMIT
git checkout $GIT_COMMIT
git clean -xdff

echo "git checkout done, t=$SECONDS"
du -hs $SOURCE_DIR $SOURCE_DIR/.git

rsync -a --delete "$SOURCE_DIR" "$TEST_DIR"

# /usr/comma/shims/uv wraps uv in sudo, which roots the venv — use the real binary
UV=$(type -ap uv | grep -vF /usr/comma/shims | head -n1)
# use panda's environment so dependencies come from its pyproject.toml
PYTHONWARNINGS=default "$UV" sync --project "$TEST_DIR" --cache-dir="/data/uv_cache" --all-extras --upgrade-package opendbc
"$UV" cache prune --cache-dir=/data/uv_cache

echo "$TEST_DIR synced with $GIT_COMMIT, t=$SECONDS"
