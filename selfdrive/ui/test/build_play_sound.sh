#!/bin/sh
clang -fPIC -o play_sound play_sound.c ../slplay.c -I ../../ -I ../ -lOpenSLES -Wl,-rpath=/system/lib64

