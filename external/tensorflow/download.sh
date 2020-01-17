#!/bin/bash
TF=libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz
#TF=libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz
#TF=libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz

if [ ! -f $TF ]; then
  wget https://storage.googleapis.com/tensorflow/libtensorflow/$TF
fi
rm -rf include lib
tar xvf $TF

