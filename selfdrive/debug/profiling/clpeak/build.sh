#!/usr/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

if [ ! -d "$DIR/clpeak" ]; then
  git clone https://github.com/krrishnarraj/clpeak.git
fi

cd clpeak
git fetch
git checkout master
git reset --hard origin/master
git submodule update --init --recursive --remote

mkdir build
cd build
cmake ..
cmake --build .
