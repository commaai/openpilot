#!/bin/bash -e
/opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ joint_matrix_bfloat16.cpp -fsycl
SYCL_PI_TRACE=1 ./a.out 
