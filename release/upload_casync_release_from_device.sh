#!/usr/bin/bash

set -ex

DEVICE=$1
CASYNC_DEVICE=/tmp/casync_device

rm -rf $CASYNC_DEVICE
scp -r $DEVICE:/data/casync $CASYNC_DEVICE
CASYNC_DIR="$CASYNC_DEVICE" release/upload_casync_release.sh
