#!/bin/bash
clang++ -I /home/batman/one/external/tensorflow/include/ -L /home/batman/one/external/tensorflow/lib -Wl,-rpath=/home/batman/one/external/tensorflow/lib main.cc -ltensorflow
