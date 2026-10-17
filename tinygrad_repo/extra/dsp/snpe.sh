#!/bin/bash -e
echo "building"
gcc -shared -fPIC -o preload_python.so preload.c -L/usr/local/pyenv/versions/3.11.4/lib -lpython3.11 -I/usr/local/pyenv/versions/3.11.4/include/python3.11
echo "compiled"
export LD_LIBRARY_PATH="/usr/local/pyenv/versions/3.11.4/lib;/data/snpe"
export LD_PRELOAD="$PWD/preload_python.so"
export PYTHONPATH="/data/tinygrad"
cd /data/snpe
#ADSP_LIBRARY_PATH="." strace -f -e ioctl ./snpe-net-run --container MobileNetV2.dlc --input_list hello --use_dsp
ADSP_LIBRARY_PATH="." ./snpe-net-run --container MobileNetV2.dlc --input_list hello --use_dsp

