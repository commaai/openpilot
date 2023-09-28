#!/usr/bin/env bash

source .devcontainer/.host/.env

# setup safe directories for submodules
SUBMODULE_DIRS=$(git config --file .gitmodules --get-regexp path | awk '{ print $2 }')
for DIR in $SUBMODULE_DIRS; do 
  git config --global --add safe.directory "$PWD/$DIR"
done

# virtual display for virtualgl
if [[ $HOST_OS == "darwin" ]]; then
  echo "Starting virtual display at :0 ..."
  tmux new-session -d -s fakedisplay Xvfb :0 -screen 0 1920x1080x24
fi
