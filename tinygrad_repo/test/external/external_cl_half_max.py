from tinygrad.runtime.ops_gpu import CLDevice, CLProgram, compile_cl

if __name__ == "__main__":
  dev = CLDevice()
  lib = compile_cl("""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void test(__global half *out, __global half *a, __global half *b) {
  int gid = get_global_id(0);
  out[gid] = max(a[gid], b[gid]);
}
""")
  prg = CLProgram(dev, "test", lib)

