#!/usr/bin/env bash
gcc -I/data/openpilot/phonelibs/opencl/include -L/system/vendor/lib64 -lOpenCL -lCB -lgsl go.c
