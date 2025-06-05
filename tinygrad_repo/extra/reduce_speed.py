import numpy as np
import ctypes
from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import lower_schedule, CompiledRunner
from tinygrad.device import CPUProgram
from dataclasses import replace
from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN

# only the memory access, over 100 GB/s! (sometimes)
reduce_asm = """
movi  v0.2d, #0000000000000000
mov   w9, #0x30
mov   w10, #0x20
mov   x8, #-0x10
movi  v1.2d, #0000000000000000
movk  w9, #0x300, lsl #16
movi  v2.2d, #0000000000000000
movk  w10, #0x200, lsl #16
movi  v3.2d, #0000000000000000
mov   w11, #0x1000000
mov   w12, #0x3ffff0
loop:
ldp   q4, q5, [x1]
add   x13, x1, x11
add   x15, x1, x10
add   x14, x1, x9
add   x8, x8, #0x10
cmp   x8, x12
ldp   q6, q7, [x1, #0x20]
add   x1, x1, #0x40
ldp   q4, q5, [x13]
ldp   q6, q7, [x13, #0x20]
ldp   q4, q5, [x15, #-0x20]
ldp   q6, q7, [x15]
ldp   q4, q5, [x14, #-0x30]
ldp   q6, q7, [x14, #-0x10]
b.lo  loop
fadd  v0.4s, v1.4s, v0.4s
fadd  v0.4s, v2.4s, v0.4s
fadd  v0.4s, v3.4s, v0.4s
dup   v1.4s, v0.s[1]
dup   v2.4s, v0.s[2]
fadd  v1.4s, v0.4s, v1.4s
dup   v0.4s, v0.s[3]
fadd  v1.4s, v2.4s, v1.4s
fadd  v0.4s, v0.4s, v1.4s
str   s0, [x0]
ret
"""

ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
arm_bytecode, _ = ks.asm(reduce_asm)
arm_bytecode = bytes(arm_bytecode)

reduce_src = """
// data1 is 16M inputs
typedef float float4 __attribute__((aligned(32),vector_size(16)));
void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc4 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc5 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc6 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc7 = {0.0f, 0.0f, 0.0f, 0.0f};
  float* data1_1 = data1+4194304;
  float* data1_2 = data1+(4194304*2);
  float* data1_3 = data1+(4194304*3);
  for (int ridx0 = 0; ridx0 < 16777216/4; ridx0+=16) {
    float4 val0 = *(float4*)((data1+(ridx0+0)));
    float4 val1 = *(float4*)((data1+(ridx0+4)));
    float4 val2 = *(float4*)((data1+(ridx0+8)));
    float4 val3 = *(float4*)((data1+(ridx0+12)));
    acc0 += val0;
    acc1 += val1;
    acc2 += val2;
    acc3 += val3;
    val0 = *(float4*)((data1_1+(ridx0+0)));
    val1 = *(float4*)((data1_1+(ridx0+4)));
    val2 = *(float4*)((data1_1+(ridx0+8)));
    val3 = *(float4*)((data1_1+(ridx0+12)));
    acc4 += val0;
    acc5 += val1;
    acc6 += val2;
    acc7 += val3;
    val0 = *(float4*)((data1_2+(ridx0+0)));
    val1 = *(float4*)((data1_2+(ridx0+4)));
    val2 = *(float4*)((data1_2+(ridx0+8)));
    val3 = *(float4*)((data1_2+(ridx0+12)));
    acc0 += val0;
    acc1 += val1;
    acc2 += val2;
    acc3 += val3;
    val0 = *(float4*)((data1_3+(ridx0+0)));
    val1 = *(float4*)((data1_3+(ridx0+4)));
    val2 = *(float4*)((data1_3+(ridx0+8)));
    val3 = *(float4*)((data1_3+(ridx0+12)));
    acc4 += val0;
    acc5 += val1;
    acc6 += val2;
    acc7 += val3;
  }
  float4 out = acc0+acc1+acc2+acc3+acc4+acc5+acc6+acc7;
  *(data0+0) = out[0]+out[1]+out[2]+out[3];
}
"""

if __name__ == "__main__":
  a = Tensor(np_array:=(np.random.default_rng().random((4096, 4096), dtype=np.float32)-0.5)).realize()
  with Context(SPLIT_REDUCEOP=0):
    # TODO: make it easy to alter the OptOps for a ScheduleItem
    GlobalCounters.reset()
    out = a.sum()
    sis = out.schedule()
    for i,(_,ei) in enumerate(lower_schedule(sis)):
      if i == 0:
        # change the source code
        prg_spec = ei.prg.p
        prg_spec = replace(prg_spec, name="reduce", src=reduce_src)
        prg = CompiledRunner(prg_spec)
        # change the assembly
        #prg._prg = CPUProgram(prg_spec.name, arm_bytecode)
        print("buffer at:",hex(ctypes.addressof(ei.bufs[1]._buf)))
        ei = replace(ei, prg=prg)
      ei.run()
    print(out.item())
    np.testing.assert_allclose(out.item(), np_array.sum(), atol=1, rtol=1e-4)
