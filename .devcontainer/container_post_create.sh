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
