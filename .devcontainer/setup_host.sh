#!/usr/bin/env bash

# setup links to Xauthority
XAUTHORITY_LINK=".devcontainer/.Xauthority"
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
