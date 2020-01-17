#!/bin/sh

gcc -DQCOM -I ~/one -I ~/one/selfdrive -I ../../include \
  -I ~/one/phonelibs/android_system_core/include -I ~/one/phonelibs/opencl/include \
  -I ~/one/selfdrive/visiond/cameras \
  test.c ../../cameras/camera_qcom.c \
  -l:libczmq.a -l:libzmq.a -lgnustl_shared -lm -llog -lcutils \
  -l:libcapn.a -l:libcapnp.a -l:libkj.a \
  ~/one/cereal/gen/c/log.capnp.o

