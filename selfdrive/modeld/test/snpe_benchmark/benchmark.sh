#!/bin/sh -e
clang++ -I ~/one/phonelibs/snpe/include/ -lSNPE -lsymphony-cpu -lsymphonypower benchmark.cc -o benchmark
./benchmark $1
