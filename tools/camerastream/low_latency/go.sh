#!/bin/bash
clang++ main.cc \
  -I/usr/local/cuda/include -I/home/batman/Downloads/Video_Codec_SDK_11.1.5/Interface \
  -o low_latency \
  -lcuda -lnvcuvid \
  -I/home/batman/openpilot /home/batman/openpilot/cereal/libmessaging.a \
  -lzmq -lcapnp -lkj -I/usr/include/SDL2 -lSDL2 -lOpenGL

