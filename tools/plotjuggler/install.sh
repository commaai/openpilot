#!/bin/bash -e

PLATFORM=$(uname)
check_platform() {
  if [[ $PLATFORM == Linux ]]; then
    echo "[PASS] PLATFORM: $PLATFORM"
  else
    echo "[FAIL] PLATFORM: $PLATFORM"
    echo
    echo "Today, plotjuggler can only be set up on Linux. Tomorrow, with the"
    echo "help of your generous contribution, we may support more platforms."
    echo 
    exit 1
  fi
}
check_platform

mkdir -p bin
cd bin

for lib_name in libDataLoadRlog.so libDataStreamCereal.so plotjuggler; do
  wget https://github.com/commaai/PlotJuggler/releases/download/latest/${lib_name}.tar.gz
  tar -xf ${lib_name}.tar.gz
  rm ${lib_name}.tar.gz
done
