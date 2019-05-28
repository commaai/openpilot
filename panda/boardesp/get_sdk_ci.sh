#!/bin/bash
git clone --recursive https://github.com/pfalcon/esp-open-sdk.git
cd esp-open-sdk
git checkout 03f5e898a059451ec5f3de30e7feff30455f7cec
LD_LIBRARY_PATH="" make STANDALONE=y
