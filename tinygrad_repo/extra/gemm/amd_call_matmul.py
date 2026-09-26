from tinygrad import UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.dtype import AddrSpace, dtypes

# open ranges around a function body: they are passed as the leading args and ended on the return value.
# values from enclosing scopes are captured by the normal python closure over the UOps.
def call(*ranges):
  def decorator(fxn):
    def wrapper(*args):
      ret = fxn(*ranges, *args).end(*ranges)
      # closing the GLOBAL ranges finishes the kernel
      if any(r.arg[-1] is AxisType.GLOBAL for r in ranges): ret = ret.sink(arg=KernelInfo(opts_to_apply=()))
      return ret
    return wrapper
  return decorator

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)

WARP_SIZE = 32
BLOCK_M, BLOCK_N = 128, 128
BLOCK_K = getenv("BK", 16)
assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

WAVES_M, WAVES_N = 4, 1
LANES_PER_WAVE_M, LANES_PER_WAVE_N = 4, 8
UNROLL_M, UNROLL_N = 4, 4

# WARP_SIZE * total waves
THREADS_PER_BLOCK = WARP_SIZE * WAVES_M * WAVES_N

# accumulator size
TM = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M)
TN = BLOCK_N // (WAVES_N * LANES_PER_WAVE_N)

@call(UOp.range(WARP_SIZE, -1, AxisType.WARP),
      UOp.range(WAVES_M, 2, AxisType.LOCAL),
      UOp.range(WAVES_N, 3, AxisType.LOCAL))
def block_128x128_gemm(lane:UOp, wave_m:UOp, wave_n:UOp, c:UOp, a:UOp, b:UOp) -> UOp:
  tid = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane

  # split global for tile reduce
  a = a.reshape(K // BLOCK_K, BLOCK_K, BLOCK_M)
  b = b.reshape(K // BLOCK_K, BLOCK_K, BLOCK_N)

  # accumulator (unified: both paths use (TM, TN) with scalar dtypes.float)
  acc = UOp.placeholder((TM, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.zeros_like(buffer=False)))

  @call(UOp.range(K // BLOCK_K, 100, AxisType.REDUCE))
  def tile_reduce(k_tile:UOp):
    A_local = UOp.placeholder((BLOCK_K, BLOCK_M), a.dtype, slot=0, addrspace=AddrSpace.LOCAL)
    B_local = UOp.placeholder((BLOCK_K, BLOCK_N), b.dtype, slot=1, addrspace=AddrSpace.LOCAL)

    # copy global -> local
    A_store = A_local.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(a[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])
    B_store = B_local.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(b[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])

    # NOTE: no explicit barrier needed, the AFTER on the LOCAL buffers implies it in late codegen
    A_local, B_local = A_local.after(A_store, B_store), B_local.after(A_store, B_store)

    @call(UOp.range(BLOCK_K, 101, AxisType.REDUCE))
    def inner_reduce(k:UOp):
      # registers for LOCAL -> REG
      a_frag = UOp.placeholder((TM//UNROLL_M, UNROLL_M), dtypes.float, slot=0, addrspace=AddrSpace.REG)
      b_frag = UOp.placeholder((TN//UNROLL_N, UNROLL_N), dtypes.float, slot=1, addrspace=AddrSpace.REG)

      # copy from local -> reg
      lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
      a_frag = a_frag.after(a_frag.store(A_local[k].reshape(WAVES_M, TM//UNROLL_M, LANES_PER_WAVE_M, UNROLL_M)[wave_m, :, lane_m, :]))
      b_frag = b_frag.after(b_frag.store(B_local[k].reshape(WAVES_N, TN//UNROLL_N, LANES_PER_WAVE_N, UNROLL_N)[wave_n, :, lane_n, :]))

      # FMA
      a_frag = a_frag.reshape(TM, 1).expand(TM, TN)
      b_frag = b_frag.reshape(1, TN).expand(TM, TN)
      # NOTE: acc.after(k) makes the accumulator load loop-carried on k
      return acc.store(acc.after(k) + (a_frag * b_frag))

    # NOTE: no explicit barrier needed, the AFTER on the LOCAL buffers implies it in late codegen
    return inner_reduce()

  # run the matmul
  acc = acc.after(tile_reduce())

  # store accumulator to output (unified)
  c = c.reshape(WAVES_M, TM//UNROLL_M, LANES_PER_WAVE_M, UNROLL_M,
                WAVES_N, TN//UNROLL_N, LANES_PER_WAVE_N, UNROLL_N)
  c = c.permute((0,4,2,6, 1,3,5,7)).reshape(THREADS_PER_BLOCK, TM, TN)
  return c[tid].store(acc)

@call(UOp.range(M // BLOCK_M, 0, AxisType.GLOBAL),
      UOp.range(N // BLOCK_N, 1, AxisType.GLOBAL))
def amd_copy_matmul(block_id_m:UOp, block_id_n:UOp, c:UOp, a:UOp, b:UOp) -> UOp:
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, block_id_m, :]
  b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, block_id_n, :]
  return block_128x128_gemm(c, a, b)

if __name__ == "__main__":
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(amd_copy_matmul, dtypes.float)



