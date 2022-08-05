#!/usr/bin/env sh
cd /tmp
git clone --recursive https://github.com/commaai/mapbox-gl-native.git
cd mapbox-gl-native
mkdir build && cd build
cmake -DMBGL_WITH_QT=ON ..
make -j$(nproc) mbgl-qt
