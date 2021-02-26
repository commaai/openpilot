#!/bin/bash -e

wget https://github.com/commaai/PlotJuggler/releases/download/latest/libDataLoadRlog.so.tar.gz
tar -xf libDataLoadRlog.so.tar.gz
rm libDataLoadRlog.so.tar.gz

wget https://github.com/commaai/PlotJuggler/releases/download/latest/plotjuggler.tar.gz
tar -xf plotjuggler.tar.gz
rm plotjuggler.tar.gz

mkdir -p bin
mv libDataLoadRlog.so bin
mv plotjuggler bin
