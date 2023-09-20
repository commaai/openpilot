#!/usr/bin/env bash

# pull base image
docker pull ghcr.io/commaai/openpilot-base:latest

# setup .host dir
mkdir -p .devcontainer/.host

# setup links to Xauthority
XAUTHORITY_LINK=".devcontainer/.host/.Xauthority"
rm -f $XAUTHORITY_LINK
if [[ -z $XAUTHORITY ]]; then
  echo "XAUTHORITY not set. Fallback to ~/.Xauthority ..."
  if ! [[ -f $HOME/.Xauthority ]]; then
    echo "~/.XAuthority file does not exist. GUI tools may not work properly."
    touch $XAUTHORITY_LINK # dummy file to satisfy container volume mount
  else
    ln -sf $HOME/.Xauthority $XAUTHORITY_LINK
  fi
else
    ln -sf $XAUTHORITY $XAUTHORITY_LINK
fi

# setup host env file
HOST_INFO_FILE=".devcontainer/.host/.env"
SYSTEM=$(uname -s | tr '[:upper:]' '[:lower:]')
echo "HOST_OS=\"$SYSTEM\"" > $HOST_INFO_FILE
