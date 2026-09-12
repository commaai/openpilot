#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit

if [ ! -d "$DIR/clpeak" ]; then
  git clone https://github.com/krrishnarraj/clpeak.git

  cd clpeak || exit
  git fetch
  git checkout ec2d3e70e1abc7738b81f9277c7af79d89b2133b
  git reset --hard origin/master
  git submodule update --init --recursive --remote

  git apply ../run_continuously.patch
fi

cd "$DIR/clpeak" || exit
mkdir build || true
cd build || exit
cmake ..
cmake --build .
