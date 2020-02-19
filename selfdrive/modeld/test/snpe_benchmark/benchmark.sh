#!/bin/sh -e
clang++ -I /data/openpilot/phonelibs/snpe/include/ -lSNPE -lsymphony-cpu -lsymphonypower benchmark.cc -o benchmark
./benchmark $1
