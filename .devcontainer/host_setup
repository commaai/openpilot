#!/usr/bin/env bash

# pull base image
if [[ -z $USE_LOCAL_IMAGE ]]; then
  echo "Updating openpilot_base image if needed..."
  docker pull ghcr.io/commaai/openpilot-base:latest
fi

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
echo "HOST_DISPLAY=\"$DISPLAY\"" >> $HOST_INFO_FILE

# run virtualgl if macos
if [[ $SYSTEM == "darwin" ]]; then
  echo
  if [[ -f /opt/VirtualGL/bin/vglclient ]]; then
    echo "Starting VirtualGL client at port 10000..."
    VGL_LOG_FILE=".devcontainer/.host/.vgl/vglclient.log"
    mkdir -p "$(dirname $VGL_LOG_FILE)"
    /opt/VirtualGL/bin/vglclient -l "$VGL_LOG_FILE" -display "$DISPLAY" -port 10000 -detach
  else
    echo "VirtualGL not found. GUI tools may not work properly. Some GUI tools require OpenGL to work properly. To use them with XQuartz on mac, VirtualGL needs to be installed. To install it run:"
    echo
    echo "  brew install --cask virtualgl"
    echo
  fi
fi
