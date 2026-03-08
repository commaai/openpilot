import os, pathlib

# TODO: there is a timing bug without this
os.environ["AMD_AQL"] = "1"

from tinygrad.device import Device
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from extra.assembly.amd.dsl import Reg, Inst, s, v

NUM_WORKGROUPS = 96
WAVE_SIZE = 32
NUM_WAVES = 2
FLOPS_PER_MATMUL = 16*16*16*2
INTERNAL_LOOP = 1_000_00
INSTRUCTIONS_PER_LOOP = 200
DIRECTIVE = ".amdhsa_wavefront_size32 1"

assemblyTemplate = (pathlib.Path(__file__).parent / "template.s").read_text()

def repeat(insts:list[Inst], n:int, counter_sreg:Reg) -> bytes:
  preamble = s_mov_b32(counter_sreg, n).to_bytes()
  insts_bytes = b"".join([inst.to_bytes() for inst in insts])
  sub_inst, cmp_inst = s_sub_u32(counter_sreg, counter_sreg, 1), s_cmp_lg_i32(counter_sreg, 0)
  loop_sz = len(insts_bytes) + sub_inst.size() + cmp_inst.size()
  branch_inst = s_cbranch_scc1(simm16=-((loop_sz // 4) + 1) & 0xFFFF)
  return preamble + insts_bytes + sub_inst.to_bytes() + cmp_inst.to_bytes() + branch_inst.to_bytes() + s_endpgm().to_bytes()

def launchBenchmark(instruction, vgprIndices, dense=True, accum=False, **kwargs):
  if accum:
    inst = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], 1, acc_cd=1, **kwargs)
  elif dense:
    inst = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], 1)
  else:
    inst = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[3]:vgprIndices[4]], v[vgprIndices[5]])
  vgprs:set = set()
  for n,_ in inst._fields:
    if isinstance(val:=getattr(inst, n), Reg) and val.offset >= v.offset: vgprs |= {val.offset+i for i in range(val.sz)}
  inst_bytes = repeat([inst for _ in range(INSTRUCTIONS_PER_LOOP)], n=INTERNAL_LOOP, counter_sreg=s[1])
  inst_hex = "\n".join("  .byte " + ",".join(f"0x{b:02x}" for b in inst_bytes[i:i+16]) for i in range(0, len(inst_bytes), 16)) + "\n"
  src = assemblyTemplate.replace("INTERNAL_LOOP", str(INTERNAL_LOOP)).replace("INSTRUCTION", inst_hex).replace("VGPR_COUNT", str(len(vgprs)))
  src = src.replace("DIRECTIVE", DIRECTIVE)
  lib = COMPILER.compile(src)
  fxn = DEV.runtime("matmul", lib)
  elapsed = min([fxn(global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True) for _ in range(2)])
  FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP * INSTRUCTIONS_PER_LOOP
  print(f"{inst.op_name.lower():<29} : {FLOPs/elapsed/10**12:.2f} T(FL)OPS")

if __name__=="__main__":
  DEV = Device[Device.DEFAULT]
  arch = DEV.renderer.arch

  COMPILER = HIPCompiler(arch)
  if arch in {'gfx1100', 'gfx1103', 'gfx1151'}:
    from extra.assembly.amd.autogen.rdna3.ins import *
    if arch == 'gfx1103': NUM_WORKGROUPS = 8
    if arch == 'gfx1151': NUM_WORKGROUPS = 32
    launchBenchmark(v_wmma_bf16_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f16_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_i32_16x16x16_iu4, (7,8,9))
    launchBenchmark(v_wmma_i32_16x16x16_iu8, (7,8,11))
  elif arch in {'gfx1200', 'gfx1201'}:
    from extra.assembly.amd.autogen.rdna4.ins import *
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
    from extra.assembly.amd.autogen.cdna.ins import *
    DIRECTIVE = ".amdhsa_accum_offset 4"
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
