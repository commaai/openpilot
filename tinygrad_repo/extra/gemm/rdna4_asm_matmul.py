# RDNA4 128x128 GEMM using WMMA — optimized DS scheduling
import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv, colored
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.engine.realize import Estimates, run_linear
from tinygrad.renderer.amd.dsl import s, v, VCC_LO, NULL, src, ttmp
from tinygrad.runtime.autogen.amd.rdna4.ins import *

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 16
TILES_M, TILES_N = 4, 4
THREADS, ELEM = 128, 2
LDS_A_ROW = BLOCK_K*ELEM   # 32
LDS_B_ROW = BLOCK_N*ELEM  # 256
LDS_A_SIZE = BLOCK_M * LDS_A_ROW  # 4096
LDS_B_SIZE = BLOCK_K * LDS_B_ROW  # 4096
LDS_SIZE = LDS_A_SIZE + LDS_B_SIZE  # 8192
LDS_B_OFF = LDS_A_SIZE
ACC, DA, DB, FA, FB, ET = 60, 188, 196, 204, 44, 10

def build_kernel(N, arch='gfx1200'):
  assert N % BLOCK_M == 0 and N >= 256
  NO_ALU, NO_DS, NO_GLOBAL = getenv("NO_ALU", 0), getenv("NO_DS", 0), getenv("NO_GLOBAL", 0)
  I, L, B = [], {}, []
  def e(i): I.append(i); return i
  def label(n): L[n] = sum(i.size() for i in I)
  def br(i, t): B.append((len(I)-1, t))

  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], ioffset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], ioffset=0x10, soffset=NULL))
  e(s_wait_kmcnt(simm16=0))
  e(s_mov_b32(s[10], ttmp[9])); e(s_and_b32(s[11], ttmp[7], 0xFFFF))
  e(s_lshl_b32(s[10], s[10], 7)); e(s_lshl_b32(s[11], s[11], 7))
  e(s_mov_b32(s[12], N)); e(s_lshl_b32(s[13], s[12], 1))
  e(s_mul_i32(s[14], s[12], BLOCK_K*ELEM))
  e(s_add_co_i32(s[17], s[12], -2*BLOCK_K))  # loop bound

  e(v_and_b32_e32(v[1], 31, v[0])); e(v_lshrrev_b32_e32(v[2], 5, v[0]))
  e(v_and_b32_e32(v[3], 1, v[2])); e(v_lshrrev_b32_e32(v[2], 1, v[2]))

  e(v_lshlrev_b32_e32(v[4], 5, v[0]))
  # B store: transposed layout for stride-32 reads. addr = LDS_B_OFF + (tid%8)*512 + (tid/8)*32
  e(v_and_b32_e32(v[48], 7, v[0])); e(v_lshlrev_b32_e32(v[5], 9, v[48]))   # (tid%8)*512
  e(v_lshrrev_b32_e32(v[48], 3, v[0])); e(v_lshlrev_b32_e32(v[48], 5, v[48]))  # (tid/8)*32
  e(v_add_nc_u32_e32(v[5], v[5], v[48])); e(v_add_nc_u32_e32(v[5], LDS_B_OFF, v[5]))

  e(v_add_nc_u32_e32(v[48], s[11], v[0]))
  e(v_mul_lo_u32(v[6], v[48], N*ELEM)); e(v_mov_b32_e32(v[7], 0))
  e(v_lshrrev_b32_e32(v[48], 3, v[0])); e(v_mul_lo_u32(v[8], v[48], N*ELEM))
  e(v_and_b32_e32(v[48], 7, v[0])); e(v_lshlrev_b32_e32(v[48], 5, v[48]))
  e(v_add_nc_u32_e32(v[8], v[8], v[48]))
  e(s_mul_i32(s[15], s[10], ELEM)); e(v_add_nc_u32_e32(v[8], s[15], v[8]))
  e(v_mov_b32_e32(v[9], 0))

  # LDS read addrs with padded strides (eliminates bank conflicts)
  # A: (lane%16)*LDS_A_ROW + (lane/16)*16 + wave_m*64*LDS_A_ROW
  # B: (lane%16)*LDS_B_ROW + (lane/16)*16 + wave_n*64*ELEM + LDS_B_OFF
  LLA, LLB = 40, 43
  e(v_and_b32_e32(v[50], 15, v[1])); e(v_lshrrev_b32_e32(v[51], 4, v[1]))
  e(v_lshlrev_b32_e32(v[LLA], 5, v[50]))        # (lane%16) * 32
  e(v_lshlrev_b32_e32(v[51], 4, v[51]))        # (lane/16) * 16
  e(v_add_nc_u32_e32(v[LLA], v[LLA], v[51]))
  e(v_lshlrev_b32_e32(v[52], 11, v[2]))         # wave_m * 2048
  e(v_add_nc_u32_e32(v[LLA], v[LLA], v[52]))
  # B read: transposed layout. addr = LDS_B_OFF + (lane%16)*32 + (lane/16)*16 + wave_n*2*512
  # wave_n selects column panels: wave_n*2 panels (each panel=16 cols, wave_n covers 64 cols = 4 panels)
  # But wave_n*2*512 = wave_n*1024. Hmm, wave_n covers cols [wave_n*64 : (wave_n+1)*64].
  # Each panel = 16 cols = 512 bytes. wave_n*64/16 = wave_n*4 panels. Offset = wave_n*4*512 = wave_n*2048.
  e(v_lshlrev_b32_e32(v[LLB], 5, v[50]))         # (lane%16) * 32 (stride 32!)
  e(v_add_nc_u32_e32(v[LLB], v[LLB], v[51]))     # + (lane/16)*16
  e(v_lshlrev_b32_e32(v[52], 11, v[3]))           # wave_n * 2048
  e(v_add_nc_u32_e32(v[LLB], v[LLB], v[52]))
  e(v_add_nc_u32_e32(v[LLB], LDS_B_OFF, v[LLB]))

  for i in range(0, 128, 2):
    e(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[ACC+i], vdsty=v[ACC+i+1], srcx0=0, srcy0=0))
  e(s_mov_b32(s[16], 0))

  if not NO_GLOBAL:
    for i in range(2): e(global_load_b128(vdst=v[DA+i*4:DA+i*4+3], vaddr=v[6:7], saddr=s[4:5], ioffset=i*16))
    for i in range(2): e(global_load_b128(vdst=v[DB+i*4:DB+i*4+3], vaddr=v[8:9], saddr=s[6:7], ioffset=i*16))
    e(s_wait_loadcnt(simm16=0))
  if not NO_DS:
    for i in range(2): e(ds_store_b128(addr=v[4], data0=v[DA+i*4:DA+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
    for i in range(2): e(ds_store_b128(addr=v[5], data0=v[DB+i*4:DB+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
  if not NO_GLOBAL:
    e(v_add_nc_u32_e32(v[6], BLOCK_K*ELEM, v[6]))
    e(v_add_nc_u32_e32(v[8], s[14], v[8]))

  # =============================================================================
  def emit_iter_body(load_set='AB'):
    if not NO_DS:
      e(s_wait_dscnt(simm16=0))
      e(s_barrier_signal(ssrc0=src[193])); e(s_barrier_wait(simm16=0xFFFF))
    if not NO_GLOBAL:
      if 'A' in load_set:
        for i in range(2): e(global_load_b128(vdst=v[DA+i*4:DA+i*4+3], vaddr=v[6:7], saddr=s[4:5], ioffset=i*16))
        e(v_add_nc_u32_e32(v[6], BLOCK_K*ELEM, v[6]))
      if 'B' in load_set:
        for i in range(2): e(global_load_b128(vdst=v[DB+i*4:DB+i*4+3], vaddr=v[8:9], saddr=s[6:7], ioffset=i*16))
        e(v_add_nc_u32_e32(v[8], s[14], v[8]))
    if not NO_DS:
      # Issue 6 loads: A[0:3] + B[0] + B[1]. B[2:3] interleaved with WMMAs.
      for tm in range(TILES_M):
        aoff = tm * 16 * LDS_A_ROW
        e(ds_load_b128(vdst=v[FA+tm*4:FA+tm*4+3], addr=v[LLA], offset0=aoff&0xFF, offset1=aoff>>8))
      e(ds_load_b128(vdst=v[FB:FB+3], addr=v[LLB], offset0=0, offset1=0))
      e(ds_load_b128(vdst=v[FB+4:FB+7], addr=v[LLB], offset0=0, offset1=2))
      e(s_wait_dscnt(simm16=0))  # wait for 6 loads (no stall!)
    if not NO_ALU:
      # B[0] WMMAs — issue B[2] during compute
      if not NO_DS: e(ds_load_b128(vdst=v[FB+8:FB+11], addr=v[LLB], offset0=0, offset1=4))
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+0)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB:FB+3], src2=v[ac:ac+7]))
      # B[1] WMMAs — issue B[3] during compute
      if not NO_DS:
        e(ds_load_b128(vdst=v[FB+12:FB+15], addr=v[LLB], offset0=0, offset1=6))
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+1)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB+4:FB+7], src2=v[ac:ac+7]))
      # B[2] WMMAs — B[2] loaded during B[0] WMMAs (~100 cycles ago)
      if not NO_DS: e(s_wait_dscnt(simm16=1))  # B[2] done, B[3] may still be loading
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+2)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB+8:FB+11], src2=v[ac:ac+7]))
      # B[3] WMMAs
      if not NO_DS: e(s_wait_dscnt(simm16=0))
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+3)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB+12:FB+15], src2=v[ac:ac+7]))
    if not NO_GLOBAL and not NO_DS: e(s_wait_loadcnt(simm16=0))
    if not NO_DS:
      for i in range(2): e(ds_store_b128(addr=v[4], data0=v[DA+i*4:DA+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
      for i in range(2): e(ds_store_b128(addr=v[5], data0=v[DB+i*4:DB+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
    e(s_add_co_i32(s[16], s[16], BLOCK_K))

  label('LOOP')
  emit_iter_body(load_set='A')
  emit_iter_body(load_set='B')
  e(s_cmp_lt_i32(s[16], s[17])); e(s_cbranch_scc1(simm16=0)); br(I[-1], 'LOOP')

  emit_iter_body(load_set='AB')  # tail with prefetch

  # Final iteration: no prefetch, no ds_store needed
  if not NO_DS:
    e(s_wait_dscnt(simm16=0))
    e(s_barrier_signal(ssrc0=src[193])); e(s_barrier_wait(simm16=0xFFFF))
  if not NO_DS:
    for tm in range(TILES_M):
      aoff = tm * 16 * LDS_A_ROW
      e(ds_load_b128(vdst=v[FA+tm*4:FA+tm*4+3], addr=v[LLA], offset0=aoff&0xFF, offset1=aoff>>8))
    e(ds_load_b128(vdst=v[FB:FB+3], addr=v[LLB], offset0=0, offset1=0))
    e(ds_load_b128(vdst=v[FB+4:FB+7], addr=v[LLB], offset0=0, offset1=2))
    e(s_wait_dscnt(simm16=0))
  if not NO_ALU:
    if not NO_DS: e(ds_load_b128(vdst=v[FB+8:FB+11], addr=v[LLB], offset0=0, offset1=4))
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+0)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB:FB+3], src2=v[ac:ac+7]))
    if not NO_DS: e(ds_load_b128(vdst=v[FB+12:FB+15], addr=v[LLB], offset0=0, offset1=6))
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+1)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB+4:FB+7], src2=v[ac:ac+7]))
    if not NO_DS: e(s_wait_dscnt(simm16=1))
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+2)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB+8:FB+11], src2=v[ac:ac+7]))
    if not NO_DS: e(s_wait_dscnt(simm16=0))
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+3)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*4:FA+tm*4+3], src1=v[FB+12:FB+15], src2=v[ac:ac+7]))

  label('EPILOGUE')
  e(v_and_b32_e32(v[ET], 15, v[1]))
  e(v_lshrrev_b32_e32(v[ET+1], 4, v[1])); e(v_lshlrev_b32_e32(v[ET+1], 3, v[ET+1]))
  e(v_lshlrev_b32_e32(v[ET+2], 6, v[2])); e(v_add_nc_u32_e32(v[ET+2], s[11], v[ET+2]))
  e(v_lshlrev_b32_e32(v[ET+3], 6, v[3])); e(v_add_nc_u32_e32(v[ET+3], s[10], v[ET+3]))
  e(v_add_nc_u32_e32(v[ET+3], v[ET+3], v[ET])); e(v_mov_b32_e32(v[ET+5], 0))

  for tm in range(TILES_M):
    for tn in range(TILES_N):
      ac = ACC + (tm*TILES_N+tn)*8; r_off, c_off = tm*16, tn*16
      e(v_add_nc_u32_e32(v[ET+6], r_off, v[ET+2])); e(v_add_nc_u32_e32(v[ET+6], v[ET+1], v[ET+6]))
      e(v_mul_lo_u32(v[ET+4], v[ET+6], s[12])); e(v_add_nc_u32_e32(v[ET+4], v[ET+4], v[ET+3]))
      if c_off: e(v_add_nc_u32_e32(v[ET+4], c_off, v[ET+4]))
      e(v_lshlrev_b32_e32(v[ET+4], 1, v[ET+4]))
      for elem in range(8):
        e(v_cvt_f16_f32_e32(v[ET+7], v[ac+elem]))
        e(global_store_b16(vaddr=v[ET+4:ET+5], vsrc=v[ET+7], saddr=s[8:9]))
        if elem < 7: e(v_add_nc_u32_e32(v[ET+4], s[13], v[ET+4]))

  e(s_wait_storecnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())

  for idx, target in B:
    off = (L[target] - sum(i.size() for i in I[:idx+1])) // 4
    assert -32768 <= off <= 32767; I[idx].simm16 = off
  return I

N = getenv("N", 4096)

def test_matmul():
  dev = Device[Device.DEFAULT]
  arch = getattr(dev.renderer, 'arch', 'gfx1200')
  print(f"Device arch: {arch}")
  insts = build_kernel(N, arch)

  rng = np.random.default_rng(42)
  a = Tensor(rng.random((N, N), dtype=np.float32).astype(np.float16))
  b = Tensor(rng.random((N, N), dtype=np.float32).astype(np.float16))
  c = Tensor.empty(N, N, dtype=dtypes.half)
  Tensor.realize(a, b, c)

  grid, local = (N//BLOCK_N, N//BLOCK_M, 1), (THREADS, 1, 1)
  print(f"Grid: {grid}, Local: {local}")

  dname = Device.DEFAULT
  def asm_kernel(A, B, C):
    gidxs = [UOp.special(n, f"gidx{i}") for i,n in enumerate(grid)]
    lidxs = [UOp.special(THREADS, "lidx0")]
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=max(LDS_SIZE, 65536//getenv("LIMIT_OCC",2)), addrspace=AddrSpace.LOCAL), (), 'lds')
    sink = UOp.sink(A.base, B.base, C.base, lds, *gidxs, *lidxs,
                    arg=KernelInfo(name=colored("kernel","cyan"), estimates=Estimates(ops=N*N*N*2, mem=N*N*2*3)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

  c = Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2]
  linear = c.schedule_linear()

  ets = []
  with Context(DEBUG=2):
    for _ in range(getenv("CNT", 5)):
      start = GlobalCounters.time_sum_s
      run_linear(linear)
      ets.append(GlobalCounters.time_sum_s - start)
  print(f"REAL TFLOPS {N*N*N*2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    c_np = c.float().numpy()
    a_np, b_np = a.float().numpy(), b.float().numpy()
    ref = a_np @ b_np
    err = np.sqrt(np.mean((c_np - ref)**2)) / np.sqrt(np.mean(ref**2))
    print(f"relative RMSE {err:.6f}")
    if err != err or err > 0.05: raise RuntimeError(f"matmul is wrong! RMSE={err}")

if __name__ == "__main__":
  test_matmul()
