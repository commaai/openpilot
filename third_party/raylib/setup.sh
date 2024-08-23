#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d raylib_repo ]; then
  git clone https://github.com/raysan5/raylib.git raylib_repo
fi

cd raylib_repo
git fetch --tags origin 5.0
git checkout 5.0

cmake .
make -j$(nproc)
