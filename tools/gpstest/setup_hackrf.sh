#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

if [ ! -d gps-sdr-sim ]; then
  git clone https://github.com/osqzss/gps-sdr-sim.git
  cd gps-sdr-sim
  make
  cd ..
fi

if [ ! -d hackrf ]; then
  git clone https://github.com/greatscottgadgets/hackrf.git
  cd hackrf/host
  git apply ../../patches/hackrf.patch
  cmake .
  make
fi

