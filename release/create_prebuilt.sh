#!/usr/bin/bash -e

# runs on tici to create a prebuilt version of a release

set -ex

BUILD_DIR=$1

cd $BUILD_DIR

# Build
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
