from tinygrad import Device, UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)

WARP_SIZE = 32
BLOCK_M, BLOCK_N = 128, 128
BLOCK_K = getenv("BK", 16)
assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

use_wmma = getenv("WMMA")
if use_wmma:
  is_rdna4 = Device[Device.DEFAULT].renderer.target.arch.startswith("gfx12")

  WAVES_M, WAVES_N = 2, 2
  LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 16

  # wmma params
  WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
  WMMA_ACC = WMMA_M // LANES_PER_WAVE_M
  UNROLL_M, UNROLL_N = (WMMA_ACC, 1) if is_rdna4 else (1, 1)
else:
  WAVES_M, WAVES_N = 4, 1
  LANES_PER_WAVE_M, LANES_PER_WAVE_N = 4, 8
  UNROLL_M, UNROLL_N = 4, 4

# total lanes must be the warp size
assert LANES_PER_WAVE_M*LANES_PER_WAVE_N == WARP_SIZE

# WARP_SIZE * total waves
THREADS_PER_BLOCK = WARP_SIZE * WAVES_M * WAVES_N

# accumulator size
TM = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M)
TN = BLOCK_N // (WAVES_N * LANES_PER_WAVE_N)

def block_128x128_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
  wave_m = UOp.range(WAVES_M, 2, AxisType.LOCAL)
  wave_n = UOp.range(WAVES_N, 3, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane

  # -- GLOBAL -> LOCAL --
  # wmma: spatial outer, k inner (k contiguous for vectorized WMMA tile loads)
  # gemm: k outer, spatial inner
  A_local = UOp.placeholder((BLOCK_M, BLOCK_K) if use_wmma else (BLOCK_K, BLOCK_M), a.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)
  B_local = UOp.placeholder((BLOCK_N, BLOCK_K) if use_wmma else (BLOCK_K, BLOCK_N), b.dtype.base, slot=1, addrspace=AddrSpace.LOCAL)

  a = a.reshape(K // BLOCK_K, BLOCK_K, BLOCK_M)
  b = b.reshape(K // BLOCK_K, BLOCK_K, BLOCK_N)
  k_tile = UOp.range(K // BLOCK_K, 100, AxisType.REDUCE)

  # copy with transpose for wmma (input is k×spatial, LDS is spatial×k)
  A_copy = A_local.permute((1,0)) if use_wmma else A_local
  B_copy = B_local.permute((1,0)) if use_wmma else B_local
  A_store = A_copy.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(a[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  B_store = B_copy.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(b[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  barrier = UOp.barrier(A_store, B_store)
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)

  # -- COMPUTE --
  lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N

  # accumulator (unified: both paths use (TM, TN) with scalar dtypes.float)
  acc = UOp.placeholder((TM, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.zeros_like()))

  if use_wmma:
    k = UOp.range(BLOCK_K // WMMA_K, 101, AxisType.REDUCE)
    tile_m = UOp.range(TM // WMMA_ACC, 200, AxisType.LOOP)
    tile_n = UOp.range(TN, 201, AxisType.LOOP)

    acc_frag = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0,2,1)[tile_m, tile_n]
    a_frag = A_local.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_K // WMMA_K, WMMA_K)[wave_m, tile_m, lane_n, k]
    b_frag = B_local.reshape(WAVES_N, TN, WMMA_N, BLOCK_K // WMMA_K, WMMA_K)[wave_n, tile_n, lane_n, k]
    if is_rdna4:
      # NOTE: since this is part of K, these 2 can be anywhere in the frags and long as a and b match
      a_frag = a_frag.reshape(2, 8)[lane_m, :]
      b_frag = b_frag.reshape(2, 8)[lane_m, :]
    wmma = UOp(Ops.SHAPED_WMMA, dtypes.float, (a_frag, b_frag, acc_frag.after(k)), arg=((16, 16, 16), 'AMD', 32))
    acc_store = acc_frag.store(wmma).end(tile_m, tile_n)
  else:
    # registers for LOCAL -> REG
    a_frag = UOp.placeholder((TM//UNROLL_M, UNROLL_M), dtypes.float, slot=0, addrspace=AddrSpace.REG)
    b_frag = UOp.placeholder((TN//UNROLL_N, UNROLL_N), dtypes.float, slot=1, addrspace=AddrSpace.REG)

    k = UOp.range(BLOCK_K, 101, AxisType.REDUCE)
    a_frag = a_frag.after(a_frag.store(A_local[k].reshape(WAVES_M, TM//UNROLL_M, LANES_PER_WAVE_M, UNROLL_M)[wave_m, :, lane_m, :]))
    b_frag = b_frag.after(b_frag.store(B_local[k].reshape(WAVES_N, TN//UNROLL_N, LANES_PER_WAVE_N, UNROLL_N)[wave_n, :, lane_n, :]))

    # FMA
    a_frag = a_frag.reshape(TM, 1).expand(TM, TN)
    b_frag = b_frag.reshape(1, TN).expand(TM, TN)
    acc_store = acc.store(acc.after(k) + (a_frag * b_frag))

  # store accumulator and loop
  acc = acc.after(acc_store.end(k).barrier().end(k_tile))

  # store accumulator to output (unified)
  c = c.reshape(WAVES_M, TM//UNROLL_M, LANES_PER_WAVE_M, UNROLL_M,
                WAVES_N, TN//UNROLL_N, LANES_PER_WAVE_N, UNROLL_N)
  c = c.permute((0,4,2,6, 1,3,5,7)).reshape(THREADS_PER_BLOCK, TM, TN)
  return c[tid].store(acc).end(wave_m, wave_n, lane)

def amd_copy_matmul(c:UOp, a:UOp, b:UOp) -> UOp:
  block_id_m = UOp.range(M // BLOCK_M, 0, AxisType.GLOBAL)
  block_id_n = UOp.range(N // BLOCK_N, 1, AxisType.GLOBAL)
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, block_id_m, :]
  b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, block_id_n, :]
  return block_128x128_gemm(c, a, b).end(block_id_n, block_id_m).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(amd_copy_matmul, dtypes.half if use_wmma else dtypes.float)
