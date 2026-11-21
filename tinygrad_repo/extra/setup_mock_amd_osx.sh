#!/bin/bash
INSTALL_PATH="${1:-/opt/homebrew/lib}"
if [ ! -d "$INSTALL_PATH" ]; then
    USER=$(whoami)
    echo "No path $INSTALL_PATH. Will create. Might need your password..."
    echo "You can stop now and provide any location as an argument where you want to save the library (note, that not default locations should be in LD_LIBRARY_PATH, so tinygrad can find it)."
    echo "Press any key or symbol to continue..."
    read -n 1 -s

    sudo mkdir -p "$INSTALL_PATH"
    sudo chown -R "$USER":staff "$INSTALL_PATH"
fi

# Download libamd_comgr.dylib
curl -s https://api.github.com/repos/tinygrad/amdcomgr_dylib/releases/latest | \
    jq -r '.assets[] | select(.name == "libamd_comgr.dylib").browser_download_url' | \
    xargs curl -L -o $INSTALL_PATH/libamd_comgr.dylib
