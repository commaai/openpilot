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

# setup the new batman user
USER=batman

# Link root pyenv to new batman user
unlink /home/$USER/.pyenvrc
ln -s /root/.pyenvrc /home/$USER/.pyenvrc

sudo chown -R $USER: /opt/pyenv
sudo chown -R $USER: /tmp/scons_cache