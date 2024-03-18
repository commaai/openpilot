#!/usr/bin/bash

set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export CASYNC_DIR="${CASYNC_DIR:=/tmp/casync}"
export SOURCE_DIR="$(git -C $DIR rev-parse --show-toplevel)"
export BUILD_DIR="${BUILD_DIR:=$(mktemp -d)}"

echo "Creating casync channel from $SOURCE_DIR to $CASYNC_DIR"

mkdir -p $CASYNC_DIR
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

release/copy_channel_files.sh $SOURCE_DIR $BUILD_DIR
release/create_prebuilt.sh $BUILD_DIR

cd $SOURCE_DIR
release/create_casync_channel.py $BUILD_DIR $CASYNC_DIR $OPENPILOT_CHANNEL
