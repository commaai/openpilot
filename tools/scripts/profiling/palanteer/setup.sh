#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d palanteer ]; then
  git clone https://github.com/dfeneyrou/palanteer
  pip install wheel
  sudo apt install libunwind-dev libdw-dev
fi

cd palanteer
git pull

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

pip install --force-reinstall python/dist/palanteer*.whl

cp bin/palanteer $DIR/viewer
