#!/usr/bin/env bash

cd ../opencv || exit
mkdir build && cd build || exit
cmake .. \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_INSTALL_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')" \
  -D PYTHON_EXECUTABLE="$(which python3)" \
  -D PYTHON3_EXECUTABLE="$(which python3)" \
  -D PYTHON3_INCLUDE_DIR="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
  -D PYTHON3_LIBRARY="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
  -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib/modules" \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_opencv_python3=ON \
  -D WITH_GSTREAMER=ON \
  -D WITH_FFMPEG=ON \
  -D WITH_CUDA=ON \
  -D WITH_CUBLAS=ON \
  -D WITH_CUDNN=ON \
  -D CUDA_ARCH_BIN=8.7 \
  -D OPENCV_ENABLE_NONFREE=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF

make -j"$(nproc)"
sudo make install
