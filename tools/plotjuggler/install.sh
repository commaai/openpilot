#!/bin/bash -e

sudo snap install plotjuggler
wget https://github.com/commaai/PlotJuggler/releases/download/latest/libDataLoadRlog.so.tar.gz
tar -xf libDataLoadRlog.so.tar.gz
rm libDataLoadRlog.so.tar.gz
