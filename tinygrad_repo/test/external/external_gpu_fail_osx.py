# ugh, OS X OpenCL doesn't support half
from tinygrad.runtime.ops_gpu import CLDevice, CLProgram, CLCompiler

src = """#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void max_half(__global half* data0, const __global half* data1) {
  int gidx0 = get_group_id(0);
  data0[gidx0] = max(data1[gidx0], (half)0.0);
}"""

if __name__ == "__main__":
  dev = CLDevice()
  print("created device")
  lib = CLCompiler(dev, "test").compile(src)
  print("created lib", len(lib))
  prg = CLProgram(dev, "max_half", lib)
  print("created prg")