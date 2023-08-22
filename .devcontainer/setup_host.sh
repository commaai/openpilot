#!/usr/bin/env bash

# setup links to Xauthority
XAUTHORITY_LINK=".devcontainer/.Xauthority"
if [[ -z $XAUTHORITY ]]; then
    ln -sf $HOME/.Xauthority $XAUTHORITY_LINK
else
    ln -sf $XAUTHORITY $XAUTHORITY_LINK
fi
