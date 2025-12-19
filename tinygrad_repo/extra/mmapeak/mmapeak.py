import pathlib
from tinygrad.device import Device
from tinygrad.runtime.ops_amd import AMDProgram, HIPCompiler
import time
import os

NUM_WORKGROUPS = 96
WAVE_SIZE = 32
NUM_WAVES = 2
FLOPS_PER_MATMUL = 16*16*16*2
INTERNAL_LOOP = 1_000_00
INSTRUCTIONS_PER_LOOP = 200
DIRECTIVE = ".amdhsa_wavefront_size32 1"

assemblyTemplate = (pathlib.Path(__file__).parent / "template.s").read_text()

def launchBenchmark(instruction, vgprIndices, dense=True, accum=False, extra=""):
  if accum:
    instructions = "{} a[0:{}], v[{}:{}], v[{}:{}], 1{}\n".format(instruction, vgprIndices[0],
                                                                vgprIndices[1], vgprIndices[2],
                                                                vgprIndices[1], vgprIndices[2], extra)
  elif dense:
    instructions = "{} v[0:{}], v[{}:{}], v[{}:{}], 1\n".format(instruction, vgprIndices[0],
                                                                vgprIndices[1], vgprIndices[2],
                                                                vgprIndices[1], vgprIndices[2])
  else:
    instructions = "{} v[0:{}], v[{}:{}], v[{}:{}], v{}\n".format(instruction, vgprIndices[0],
                                                                  vgprIndices[1], vgprIndices[2],
                                                                  vgprIndices[3], vgprIndices[4],
                                                                  vgprIndices[5])
  src = assemblyTemplate.replace("INTERNAL_LOOP", str(INTERNAL_LOOP)).replace("INSTRUCTION", instructions*INSTRUCTIONS_PER_LOOP)
  src = src.replace("DIRECTIVE", DIRECTIVE)
  lib = COMPILER.compile(src)
  fxn = AMDProgram(DEV, "matmul", lib)
  elapsed = fxn(global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True)
  FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP * INSTRUCTIONS_PER_LOOP
  print(f"{instruction:<29} : {FLOPs/elapsed/10**12:.2f} T(FL)OPS")

if __name__=="__main__":
  DEVICENUM = os.getenv("DEVICENUM", "0")
  try:
    DEV = Device['AMD:' + DEVICENUM]
  except:
    raise RuntimeError("Error while initiating AMD device")

  COMPILER = HIPCompiler(DEV.arch)
  if DEV.arch in {'gfx1100', 'gfx1103'}:
    if DEV.arch == 'gfx1103':
      NUM_WORKGROUPS = 8
    launchBenchmark("v_wmma_bf16_16x16x16_bf16", (7,8,15))
    launchBenchmark("v_wmma_f16_16x16x16_f16", (7,8,15))
    launchBenchmark("v_wmma_f32_16x16x16_bf16", (7,8,15))
    launchBenchmark("v_wmma_f32_16x16x16_f16", (7,8,15))
    launchBenchmark("v_wmma_i32_16x16x16_iu4", (7,8,9))
    launchBenchmark("v_wmma_i32_16x16x16_iu8", (7,8,11))
  elif DEV.arch == 'gfx1201':
    NUM_WORKGROUPS = 64
    launchBenchmark("v_wmma_bf16_16x16x16_bf16", (3,4,7))
    launchBenchmark("v_wmma_f16_16x16x16_f16", (3,4,7))
    launchBenchmark("v_wmma_f32_16x16x16_bf16", (7,8,11))
    launchBenchmark("v_wmma_f32_16x16x16_f16", (7,8,11))
    launchBenchmark("v_wmma_i32_16x16x16_iu4", (7,8,8))
    launchBenchmark("v_wmma_i32_16x16x16_iu8", (7,8,9))
    launchBenchmark("v_wmma_f32_16x16x16_fp8_fp8", (7,8,9))
    launchBenchmark("v_wmma_f32_16x16x16_fp8_bf8", (7,8,9))
    launchBenchmark("v_wmma_f32_16x16x16_bf8_fp8", (7,8,9))
    launchBenchmark("v_wmma_f32_16x16x16_bf8_bf8", (7,8,9))
    FLOPS_PER_MATMUL = 16*16*32*2
    launchBenchmark("v_wmma_i32_16X16X32_iu4", (7,8,9))
    launchBenchmark("v_swmmac_f32_16x16x32_f16", (7,8,11,12,19,20), False)
    launchBenchmark("v_swmmac_f32_16x16x32_bf16", (7,8,11,12,19,20), False)
    launchBenchmark("v_swmmac_f16_16x16x32_f16", (3,4,7,8,15,16), False)
    launchBenchmark("v_swmmac_bf16_16x16x32_bf16", (3,4,7,8,15,16), False)
    launchBenchmark("v_swmmac_i32_16x16x32_iu8", (7,8,9,10,13,14), False)
    launchBenchmark("v_swmmac_i32_16x16x32_iu4", (7,8,8,9,10,11), False)
    launchBenchmark("v_swmmac_f32_16x16x32_fp8_fp8", (7,8,9,10,13,14), False)
    launchBenchmark("v_swmmac_f32_16x16x32_fp8_bf8", (7,8,9,10,13,14), False)
    launchBenchmark("v_swmmac_f32_16x16x32_bf8_fp8", (7,8,9,10,13,14), False)
    launchBenchmark("v_swmmac_f32_16x16x32_bf8_bf8", (7,8,9,10,13,14), False)
    FLOPS_PER_MATMUL = 16*16*64*2
    launchBenchmark("v_swmmac_i32_16x16x64_iu4", (7,8,9,10,13,14), False)
  elif DEV.arch == 'gfx950':
    DIRECTIVE = ".amdhsa_accum_offset 4"
    NUM_WORKGROUPS = 256
    WAVE_SIZE = 64
    NUM_WAVES = 4
    launchBenchmark("v_mfma_f32_16x16x16_f16", (3,0,1), accum=True)
    launchBenchmark("v_mfma_f32_16x16x16_bf16", (3,0,1), accum=True)
    FLOPS_PER_MATMUL = 16*16*32*2
    launchBenchmark("v_mfma_f32_16x16x32_f16", (3,0,3), accum=True)
    launchBenchmark("v_mfma_f32_16x16x32_bf16", (3,0,3), accum=True)
    FLOPS_PER_MATMUL = 16*16*128*2
    launchBenchmark("v_mfma_f32_16x16x128_f8f6f4", (3,0,7), accum=True) # fp8
    launchBenchmark("v_mfma_f32_16x16x128_f8f6f4", (3,0,5), accum=True, extra=", cbsz:2 blgp:2") # fp6
    launchBenchmark("v_mfma_f32_16x16x128_f8f6f4", (3,0,3), accum=True, extra=", cbsz:4 blgp:4") # fp4
  else:
    raise RuntimeError(f"arch {DEV.arch} not supported.")
