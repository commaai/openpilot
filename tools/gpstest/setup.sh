#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d LimeSuite ]; then
  git clone https://github.com/myriadrf/LimeSuite.git
  cd LimeSuite
  # checkout latest version which has firmware updates available
  git checkout v20.10.0
  cp ../mcu_error.patch .
  cp ../reference_print.patch .
  git apply mcu_error.patch
  git apply reference_print.patch
  mkdir builddir && cd builddir
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j4
  cd ../..
fi

if [ ! -d LimeGPS ]; then
  git clone https://github.com/osqzss/LimeGPS.git
  cd LimeGPS

  cp ../inc_ephem_array_size.patch .
  cp ../makefile.patch .
  git apply inc_ephem_array_size.patch
  git apply makefile.patch

  make
  cd ..
fi
