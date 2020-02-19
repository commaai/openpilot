#!/bin/sh -e
clang++ -I /data/openpilot/phonelibs/snpe/include/ -L/data/pythonpath/phonelibs/snpe/aarch64 -lSNPE benchmark.cc -o benchmark
export LD_LIBRARY_PATH="/data/pythonpath/phonelibs/snpe/aarch64/:$HOME/openpilot/phonelibs/snpe/x86_64/:$LD_LIBRARY_PATH"
exec ./benchmark $1
