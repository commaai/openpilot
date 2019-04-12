#!/bin/sh
set -e

make
export LD_LIBRARY_PATH=/system/lib64:$LD_LIBRARY_PATH
exec ./ui
