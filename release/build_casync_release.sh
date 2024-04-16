#!/bin/bash
set -ex

export BUILD_DIR="${BUILD_DIR:=$(mktemp -d)}"

release/build_release.sh
release/package_casync_build.py $BUILD_DIR
