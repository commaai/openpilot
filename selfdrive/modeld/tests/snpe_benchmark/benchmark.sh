#!/bin/sh -e
clang++ -I /data/openpilot/third_party/snpe/include/ -L/data/pythonpath/third_party/snpe/aarch64 -lSNPE benchmark.cc -o benchmark
export LD_LIBRARY_PATH="/data/pythonpath/third_party/snpe/aarch64/:$HOME/openpilot/third_party/snpe/x86_64/:$LD_LIBRARY_PATH"
exec ./benchmark $1
