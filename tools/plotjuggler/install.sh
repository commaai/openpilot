#!/bin/bash -e

mkdir -p bin
cd bin

for lib_name in libDataLoadRlog.so libDataStreamCereal.so plotjuggler; do
  wget https://github.com/commaai/PlotJuggler/releases/download/latest/${lib_name}.tar.gz
  tar -xf ${lib_name}.tar.gz
  rm ${lib_name}.tar.gz
done
