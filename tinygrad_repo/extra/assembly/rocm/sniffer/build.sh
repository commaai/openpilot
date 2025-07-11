#!/bin/bash -e
clang sniff.cc -Werror -shared -fPIC -I../src/ -I../src/ROCT-Thunk-Interface/include -I../src/ROCm-Device-Libs/ockl/inc -o sniff.so -lstdc++
#AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7 LD_PRELOAD=$PWD/sniff.so /home/tiny/build/HIP-Examples/HIP-Examples-Applications/HelloWorld/HelloWorld
#AMD_LOG_LEVEL=4 LD_PRELOAD=$PWD/sniff.so $HOME/build/HIP-Examples/HIP-Examples-Applications/HelloWorld/HelloWorld
#AMD_LOG_LEVEL=5 LD_PRELOAD=$PWD/sniff.so python3 ../rdna3/asm.py
DEBUG=5 LD_PRELOAD=$PWD/sniff.so python3 ../rdna3/asm.py
#AMD_LOG_LEVEL=5 HSAKMT_DEBUG_LEVEL=7 DEBUG=5 LD_PRELOAD=$PWD/sniff.so strace -F python3 ../rdna3/asm.py
#LD_PRELOAD=$PWD/sniff.so python3 ../rdna3/asm.py
#AMD_LOG_LEVEL=4 LD_PRELOAD=$PWD/sniff.so FORWARD_ONLY=1 DEBUG=2 python3 ../../../test/test_ops.py TestOps.test_add
#AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7 LD_PRELOAD=$PWD/sniff.so rocm-bandwidth-test -s 0 -d 1 -m 1
#AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7 LD_PRELOAD=$PWD/sniff.so rocm-bandwidth-test -s 1 -d 2 -m 1
