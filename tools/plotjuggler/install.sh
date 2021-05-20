#!/bin/bash -e

mkdir -p bin
cd bin

wget https://github.com/commaai/PlotJuggler/releases/download/latest/libDataLoadRlog.so.tar.gz
tar -xf libDataLoadRlog.so.tar.gz
rm libDataLoadRlog.so.tar.gz

wget https://github.com/commaai/PlotJuggler/releases/download/latest/libDataStreamCereal.so.tar.gz
tar -xf libDataStreamCereal.so.tar.gz
rm libDataStreamCereal.so.tar.gz

wget https://github.com/commaai/PlotJuggler/releases/download/latest/plotjuggler.tar.gz
tar -xf plotjuggler.tar.gz
rm plotjuggler.tar.gz
