#!/usr/bin/bash

set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:=/tmp/casync}"
export SOURCE_DIR="$(git -C $DIR rev-parse --show-toplevel)"
export BUILD_DIR="${BUILD_DIR:=$(mktemp -d)}"

echo "Creating casync channel from $SOURCE_DIR to $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

release/copy_release_files.sh $SOURCE_DIR $BUILD_DIR
release/create_prebuilt.sh $BUILD_DIR

cd $SOURCE_DIR
release/create_casync_channel.py $BUILD_DIR $OUTPUT_DIR $RELEASE_CHANNEL
