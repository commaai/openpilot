import time
import numpy as np
from tinygrad.helpers import getenv, prod, flat_mv
from tinygrad.runtime.ops_amd import AMDAllocator, AMDDevice, AMDProgram

# AMD_LOG_LEVEL=3 ./MIOpenDriver gemm --iter 1000 --time 1 --a_w 2048 --a_h 2048 --b_w 2048
# 5.5: Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4
# 5.6: Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4
# gets ~100
# hipExtModuleLaunchKernel ( 0x0x16ccde0, 2048, 16, 1, 128, 1, 1,
# 161.60 us = 106.31 TFLOPS
# with --batch_count 8 / 1.258128 ms / (8*2048*2048*2048*2)/(1.258128)*1e-9 / 109.24 TFLOPS

# we only get ~53
# KY=2 KX=2 N=2048 python3 extra/gemm/hip_matmul.py
#   4194304    324.76 us, would be  52899.88 GFLOPS matmul, 154.98 GB/s

DEBUG = getenv("DEBUG", 0)
RAND = getenv("RAND", 0)
CNT = getenv("CNT", 128)

N = getenv("N", 4096)
KX = getenv("KX", 4)
KY = getenv("KY", 4)
assert N%(16*KX) == 0, f"N must be multiple of {16*KX}"
assert N%(16*KY) == 0, f"N must be multiple of {16*KY}"
FLOPS = N*N*N*2
BW = N*N*3*4

local_size = [32, 1, 1]
global_size = [N//(KX*16), N//(KY*16), 1]
num_threads = prod(local_size)

# Can AMDAllocator initialized as device=0 by default?
device = AMDDevice()
hipallocator = AMDAllocator(device)
a = hipallocator.alloc(N*N*4)
b = hipallocator.alloc(N*N*2)
c = hipallocator.alloc(N*N*2)
na = np.empty(N*N, np.float32)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
nc = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
hipallocator._copyin(b, memoryview(bytearray(nb)))
hipallocator._copyin(c, memoryview(bytearray(nc)))

prog_str = f"""
#define F32
typedef long unsigned int size_t;
#define half _Float16
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);
extern "C" __attribute__((global))void __attribute__((amdgpu_flat_work_group_size(1, {num_threads}))) test(float* c, half* a, half* b) {{

  const int gx = __ockl_get_group_id(0) + __ockl_get_local_id(2);
  const int gy = __ockl_get_group_id(1) + __ockl_get_local_id(3);

  const int lIdx = __ockl_get_local_id(0);
  const int lane = lIdx%16;

  c += gx*{KX*16}*{N} + gy*{KY*16} + (lIdx/16)*{N} + lane;
  a += gx*{KX*16}*{N};
  b += gy*{KY*16};

  half16 a_frag[{KX}];
  half16 b_frag[{KY}];
  #ifdef F32
    float8 c_frag[{KY}][{KX}] = {{}};
  #else
    half16 c_frag[{KY}][{KX}] = {{}};
  #endif

  for (int k = 0; k < {N}; k += 16) {{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");

    for (int ele = 0; ele < 16; ++ele) {{
      for (int x = 0; x < {KX}; x++) {{
        a_frag[x][ele] = a[(k+ele) + x*{16*N} + {N}*lane];
      }}
    }}
    for (int ele = 0; ele < 16; ++ele) {{
      for (int y = 0; y < {KY}; y++) {{
        b_frag[y][ele] = b[(k+ele)*{N} + y*16 + lane];
      }}
    }}
    for (int y = 0; y < {KY}; y++) {{
      for (int x = 0; x < {KX}; x++) {{
        #ifdef F32
          c_frag[y][x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[x], b_frag[y], c_frag[y][x]);
        #else
          c_frag[y][x] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[x], b_frag[y], c_frag[y][x], false);
        #endif
      }}
    }}
  }}

  for (int ele = 0; ele < 8; ++ele) {{
    for (int y = 0; y < {KY}; y++) {{
      for (int x = 0; x < {KX}; x++) {{
        #ifdef F32
          c[ele*{2*N} + y*16 + x*{16*N}] = c_frag[y][x][ele];
        #else
          c[ele*{2*N} + y*16 + x*{16*N}] = c_frag[y][x][ele*2];
        #endif
      }}
    }}
  }}
}}"""

if DEBUG > 1: print(prog_str)
lib = device.compiler.compile(prog_str)
prog = AMDProgram(device, "test", lib)

def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  ret = time.perf_counter() - st # NOTE: et doesn't contain the launch overhead
  if DEBUG > 0: print(f"{ret*1e6:.2f} us")
  # rerun rand
  if RAND:
    nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
    nc = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
    hipallocator._copyin(b, memoryview(bytearray(nb)))
    hipallocator._copyin(c, memoryview(bytearray(nc)))
  return et

print("global/local size", global_size, local_size, f"local_size:{prod(local_size)} total_size:{prod(global_size+local_size)}")
tm = min([timeit(lambda: prog(a, b, c, global_size=global_size, local_size=local_size, wait=True)) for _ in range(CNT)])
hipallocator._copyout(flat_mv(na.data),a)
na = na.reshape(N,N)
comp = nb.astype(np.float32) @ nc.astype(np.float32)
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
if DEBUG > 2: print(f"which  nan={np.where(np.isnan(na))} len={len(np.where(np.isnan(na))[0])}")
if DEBUG > 2: print(f"which diff={np.where(abs(na-comp) > 2e-2)} len={len(np.where(abs(na-comp) > 2e-2)[0])}")
if DEBUG > 2: print(f"which zero={np.where(abs(na) < 2e-2)} len={len(np.where(abs(na) < 2e-2)[0])}")
np.testing.assert_allclose(na, comp, atol=1e-2, rtol=1e-2)
