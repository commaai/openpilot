#!/usr/bin/env bash

set -e
set -x


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
tee "$CONTINUE_PATH" << EOF
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
chmod +x "$CONTINUE_PATH"

# a DUT that booted with a continue.sh that launches openpilot is still running it, which fights the tests
if pgrep -f launch_chffrplus.sh > /dev/null; then
  sudo systemctl stop comma
fi

# submodule package symlinks for PYTHONPATH imports on device (same as launch_chffrplus.sh)
cd "$TEST_DIR"
ln -sfn msgq_repo/msgq msgq
ln -sfn opendbc_repo/opendbc opendbc
ln -sfn rednose_repo/rednose rednose
ln -sfn teleoprtc_repo/teleoprtc teleoprtc
ln -sfn tinygrad_repo/tinygrad tinygrad

echo "$TEST_DIR ready, t=$SECONDS"
