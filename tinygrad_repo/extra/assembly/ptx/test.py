#!/usr/bin/env python3
import numpy as np
from tinygrad.runtime.ops_cuda import CUDAProgram, RawCUDABuffer

if __name__ == "__main__":
  test = RawCUDABuffer.fromCPU(np.zeros(10, np.float32))
  prg = CUDAProgram("test", """
  .version 7.8
  .target sm_86
  .address_size 64
  .visible .entry test(.param .u64 x) {
    .reg .b32       %r<2>;
    .reg .b64       %rd<3>;

    ld.param.u64    %rd1, [x];
    cvta.to.global.u64      %rd2, %rd1;
    mov.u32         %r1, 0x40000000; // 2.0 in float
    st.global.u32   [%rd2], %r1;
    ret;
  }""", binary=True)
  prg([1], [1], test)
  print(test.toCPU())

