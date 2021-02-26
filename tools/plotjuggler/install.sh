#!/bin/bash -e

mkdir -p bin

wget https://github.com/commaai/PlotJuggler/releases/download/latest/libDataLoadRlog.so.tar.gz --directory-prefix bin
tar --directory bin -xf bin/libDataLoadRlog.so.tar.gz
rm bin/libDataLoadRlog.so.tar.gz

wget https://github.com/commaai/PlotJuggler/releases/download/latest/plotjuggler.tar.gz --directory-prefix bin
tar --directory bin -xf bin/plotjuggler.tar.gz
rm bin/plotjuggler.tar.gz
