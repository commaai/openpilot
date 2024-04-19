#!/usr/bin/bash

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
SOURCE_DIR="$(git -C $DIR rev-parse --show-toplevel)"
BUILD_DIR=${1:-$(mktemp -d)}

if [ -f /TICI ]; then
  FILES_SRC="release/files_tici"
else
  FILES_SRC="release/files_pc"
fi

echo "Building openpilot into $BUILD_DIR"

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

# Copy required files to BUILD_DIR
cd $SOURCE_DIR
cp -pR --parents $(cat release/files_common) $BUILD_DIR/
cp -pR --parents $(cat $FILES_SRC) $BUILD_DIR/

# Build + cleanup
cd $BUILD_DIR
export PYTHONPATH="$BUILD_DIR"

rm -f panda/board/obj/panda.bin.signed
rm -f panda/board/obj/panda_h7.bin.signed

if [ -n "$RELEASE" ]; then
  export CERT=/data/pandaextra/certs/release
fi

scons -j$(nproc)

# Cleanup
find . -name '*.a' -delete
find . -name '*.o' -delete
find . -name '*.os' -delete
find . -name '*.pyc' -delete
find . -name 'moc_*' -delete
find . -name '__pycache__' -delete
rm -rf .sconsign.dblite Jenkinsfile release/
rm selfdrive/modeld/models/supercombo.onnx

# Mark as prebuilt release
touch prebuilt

echo "----- openpilot has been built to $BUILD_DIR -----"
