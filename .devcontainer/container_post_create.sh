#!/usr/bin/env bash

source .devcontainer/.host/.env

# override display flag for mac
if [[ $HOST_OS == darwin ]]; then
  echo "Setting up DISPLAY override for macOS..."
  cat <<EOF >> /root/.bashrc
if [ -n "\$DISPLAY" ]; then
  DISPLAY_NUM=\$(echo "\$DISPLAY" | awk -F: '{print \$NF}')
  export DISPLAY=host.docker.internal:\$DISPLAY_NUM
fi
EOF
fi

# These lines are temporary, to remain backwards compatible with old devcontainers
# that were running as root and therefore had their caches written as root
USER=batman
sudo chown -R $USER: /tmp/scons_cache
sudo chown -R $USER: /tmp/comma_download_cache