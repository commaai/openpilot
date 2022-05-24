#!/bin/bash
#clang++ main.cc \
#  -I/usr/local/cuda/include -I/home/batman/Downloads/Video_Codec_SDK_11.1.5/Interface \
#  -o low_latency \
#  -lcuda -lnvcuvid \
#  -I/home/batman/openpilot /home/batman/openpilot/cereal/libmessaging.a \
#  -lzmq -lcapnp -lkj -I/usr/include/SDL2 -lSDL2 -lOpenGL

clang++	-std=c++17 -fPIC main_qt.cc \
  -o low_latency \
   /home/batman/openpilot/common/util.cc \
   /home/batman/openpilot/common/params.cc \
   /home/batman/openpilot/selfdrive/ui/qt/qt_window.cc \
   /home/batman/openpilot/selfdrive/ui/qt/util.cc \
  -I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtWidgets \
  -I/usr/include/x86_64-linux-gnu/qt5/QtGui -I/usr/include/x86_64-linux-gnu/qt5/QtCore \
  -lQt5Core -lQt5Gui -lQt5Widgets -lQt5Gui \
  -I/home/batman/openpilot
