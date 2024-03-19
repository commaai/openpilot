#!/usr/bin/bash

set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

CASYNC_DIR="${CASYNC_DIR:=/tmp/casync}"
SOURCE_DIR="$(git -C $DIR rev-parse --show-toplevel)"
BUILD_DIR="${BUILD_DIR:=$(mktemp -d)}"

echo "Creating casync release from $SOURCE_DIR to $CASYNC_DIR"

mkdir -p $CASYNC_DIR
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

release/copy_build_files.sh $SOURCE_DIR $BUILD_DIR
release/create_prebuilt.sh $BUILD_DIR

cd $SOURCE_DIR
release/create_casync_release.py $BUILD_DIR $CASYNC_DIR $OPENPILOT_CHANNEL
