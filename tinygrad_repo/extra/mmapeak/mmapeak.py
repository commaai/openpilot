import os

# TODO: there is a timing bug without this
os.environ["AMD_AQL"] = "1"

from tinygrad import Tensor, Device, GlobalCounters, Context
from tinygrad.helpers import getenv, DEV
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.renderer.amd.dsl import Reg, Inst, s, v
from tinygrad.engine.realize import run_linear

NUM_WORKGROUPS = 96
WAVE_SIZE = 32
NUM_WAVES = 4
FLOPS_PER_MATMUL = 16*16*16*2
INTERNAL_LOOP = getenv("LOOP", 10_000)
INSTRUCTIONS_PER_LOOP = 200

def repeat(insts:list[Inst], n:int, counter_sreg:Reg) -> list[Inst]:
  insts_bytes = b"".join([inst.to_bytes() for inst in insts])
  sub_inst, cmp_inst = s_sub_u32(counter_sreg, counter_sreg, 1), s_cmp_lg_i32(counter_sreg, 0)
  loop_sz = len(insts_bytes) + sub_inst.size() + cmp_inst.size()
  branch_inst = s_cbranch_scc1(simm16=-((loop_sz // 4) + 1) & 0xFFFF)
  return [s_mov_b32(counter_sreg, n)] + insts + [sub_inst, cmp_inst, branch_inst, s_endpgm()]

def launchBenchmark(instruction, vgprIndices, dense=True, accum=False, **kwargs):
  if accum:
    inst = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], 1, acc_cd=1, **kwargs)
  elif dense:
    inst = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], 1)
  else:
    inst = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[3]:vgprIndices[4]], v[vgprIndices[5]])
  insts = repeat([inst for _ in range(INSTRUCTIONS_PER_LOOP)], n=INTERNAL_LOOP, counter_sreg=s[1])
  def fxn(A:UOp) -> UOp:
    threads = UOp.special(WAVE_SIZE * NUM_WAVES, "lidx0")
    gidx = UOp.special(NUM_WORKGROUPS, "gidx0")
    FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP * INSTRUCTIONS_PER_LOOP
    sink = UOp.sink(A.base, threads, gidx, arg=KernelInfo(inst.op.name.lower(), estimates=Estimates(ops=FLOPs, mem=0)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  dummy = Tensor.zeros(1).contiguous().realize()
  out = Tensor.custom_kernel(dummy, fxn=fxn)[0]
  linear = out.schedule_linear()
  ets = []
  with Context(DEBUG=2):
    for _ in range(2):
      start = GlobalCounters.time_sum_s
      run_linear(linear)
      ets.append(GlobalCounters.time_sum_s - start)
  elapsed = min(ets)
  FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP * INSTRUCTIONS_PER_LOOP
  print(f"{inst.op_name.lower():<29} : {FLOPs/elapsed/10**12:.2f} T(FL)OPS")

if __name__=="__main__":
  DEV = Device[Device.DEFAULT]
  arch = DEV.renderer.target.arch

  if arch in {'gfx1100', 'gfx1103', 'gfx1151'}:
    from tinygrad.runtime.autogen.amd.rdna3.ins import *
    if arch == 'gfx1103': NUM_WORKGROUPS = 8
    if arch == 'gfx1151': NUM_WORKGROUPS = 32
    launchBenchmark(v_wmma_bf16_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f16_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_i32_16x16x16_iu4, (7,8,9))
    launchBenchmark(v_wmma_i32_16x16x16_iu8, (7,8,11))
  elif arch in {'gfx1200', 'gfx1201'}:
    from tinygrad.runtime.autogen.amd.rdna4.ins import *
    # this instruction does not exist in the rdna4 isa, use the co version
    s_sub_u32 = s_sub_co_u32
    NUM_WORKGROUPS = 64
    launchBenchmark(v_wmma_bf16_16x16x16_bf16, (3,4,7))
    launchBenchmark(v_wmma_f16_16x16x16_f16, (3,4,7))
    launchBenchmark(v_wmma_f32_16x16x16_bf16, (7,8,11))
    launchBenchmark(v_wmma_f32_16x16x16_f16, (7,8,11))
    launchBenchmark(v_wmma_i32_16x16x16_iu4, (7,8,8))
    launchBenchmark(v_wmma_i32_16x16x16_iu8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_fp8_fp8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_fp8_bf8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_bf8_fp8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_bf8_bf8, (7,8,9))
    FLOPS_PER_MATMUL = 16*16*32*2
    launchBenchmark(v_wmma_i32_16x16x32_iu4, (7,8,9))
    launchBenchmark(v_swmmac_f32_16x16x32_f16, (7,8,11,12,19,20), False)
    launchBenchmark(v_swmmac_f32_16x16x32_bf16, (7,8,11,12,19,20), False)
    launchBenchmark(v_swmmac_f16_16x16x32_f16, (3,4,7,8,15,16), False)
    launchBenchmark(v_swmmac_bf16_16x16x32_bf16, (3,4,7,8,15,16), False)
    launchBenchmark(v_swmmac_i32_16x16x32_iu8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_i32_16x16x32_iu4, (7,8,8,9,10,11), False)
    launchBenchmark(v_swmmac_f32_16x16x32_fp8_fp8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_f32_16x16x32_fp8_bf8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_f32_16x16x32_bf8_fp8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_f32_16x16x32_bf8_bf8, (7,8,9,10,13,14), False)
    FLOPS_PER_MATMUL = 16*16*64*2
    launchBenchmark(v_swmmac_i32_16x16x64_iu4, (7,8,9,10,13,14), False)
  elif arch == 'gfx950':
    from tinygrad.runtime.autogen.amd.cdna.ins import *
    NUM_WORKGROUPS = 256
    WAVE_SIZE = 64
    NUM_WAVES = 4
    launchBenchmark(v_mfma_f32_16x16x16_f16, (3,0,1), accum=True)
    launchBenchmark(v_mfma_f32_16x16x16_bf16, (3,0,1), accum=True)
    FLOPS_PER_MATMUL = 16*16*32*2
    launchBenchmark(v_mfma_f32_16x16x32_f16, (3,0,3), accum=True)
    launchBenchmark(v_mfma_f32_16x16x32_bf16, (3,0,3), accum=True)
    FLOPS_PER_MATMUL = 16*16*128*2
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,7), accum=True) # fp8
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,5), accum=True, cbsz=2, blgp=2) # fp6
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,3), accum=True, cbsz=4, blgp=4) # fp4
  else:
    raise RuntimeError(f"arch {arch} not supported.")
