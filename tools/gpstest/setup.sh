#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d LimeSuite ]; then
  git clone https://github.com/myriadrf/LimeSuite.git
  cd LimeSuite
  # checkout latest version which has firmware updates available
  git checkout v20.10.0
  git apply ../patches/limeSuite/*
  mkdir builddir && cd builddir
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j4
  cd ../..
fi

if [ ! -d LimeGPS ]; then
  git clone https://github.com/osqzss/LimeGPS.git
  cd LimeGPS
  git apply ../patches/limeGPS/*
  make
  cd ..
fi
