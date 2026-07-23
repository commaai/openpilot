from tinygrad import Tensor, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo, sint, AxisType
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEBUG, getenv

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
NUM_RUNS = getenv("CNT", 5)

# ---------------------------
# launch/config constants
# ---------------------------

WARP_SIZE = 32
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
TM, TN = 4, 4
LANES_PER_WAVE_M, LANES_PER_WAVE_N = 4, 8
assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

is_kernel5 = getenv("K5", 0)
THREADS_PER_BLOCK = 128 if is_kernel5 else 256
WAVES_PER_BLOCK_N = 1 if is_kernel5 else 2
WAVES_PER_BLOCK_M = THREADS_PER_BLOCK // WARP_SIZE // WAVES_PER_BLOCK_N
REG_TILES_PER_WAVE_N = BLOCK_N // (WAVES_PER_BLOCK_N * LANES_PER_WAVE_N * TN)
REG_TILES_PER_WAVE_M = BLOCK_M // (WAVES_PER_BLOCK_M * LANES_PER_WAVE_M * TM)

assert WAVES_PER_BLOCK_M*REG_TILES_PER_WAVE_M*LANES_PER_WAVE_M*TM == BLOCK_M, "M reshape is wrong"
assert WAVES_PER_BLOCK_N*REG_TILES_PER_WAVE_N*LANES_PER_WAVE_N*TN == BLOCK_N, "N reshape is wrong"

def rngs_for_shape(shape:tuple[sint, ...], rng:int, axis_type=AxisType.LOOP): return [UOp.range(s, rng+i, axis_type) for i,s in enumerate(shape)]
def copy(dest:UOp, src:UOp, rng:int, upcast=False):
  assert dest.shape == src.shape
  rngs = rngs_for_shape(src.shape, rng, AxisType.UPCAST if upcast else AxisType.LOOP)
  return dest[*rngs].store(src[*rngs]).end(*rngs)

def hand_spec_kernel3(c:UOp, a:UOp, b:UOp) -> UOp:
  # ---------------------------
  # block indices
  # ---------------------------
  block_id_n = UOp.special(N // BLOCK_N, "gidx0")
  block_id_m = UOp.special(M // BLOCK_M, "gidx1")

  # index the output with the globals
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]

  # open the main reduction range
  k_tile_range = UOp.range(K // BLOCK_K, 0, AxisType.REDUCE)
  a = a.reshape(M // BLOCK_M, BLOCK_M, K // BLOCK_K, BLOCK_K)[block_id_m, :, k_tile_range, :]
  b = b.reshape(K // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N)[k_tile_range, :, block_id_n, :]

  # globals are no longer used, they are already in the indexes
  del block_id_m, block_id_n

  # ---------------------------
  # GLOBAL -> LOCAL (A_local, B_local)
  # ---------------------------
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

  # A: read BM x BK tiles (permute on store into locals)
  BM_A_local_stride = (BLOCK_M + 4) if is_kernel5 else BLOCK_M
  A_local = UOp.placeholder((BLOCK_K, BM_A_local_stride), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_M))
  A_local_store = copy(A_local.permute((1,0)).reshape(-1, THREADS_PER_BLOCK)[:, tid], a.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=100)

  # B: read BK x BN tiles
  B_local = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)
  B_local_store = copy(B_local.reshape(-1, THREADS_PER_BLOCK)[:, tid], b.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=200)

  # TODO: can we automate barrier?
  barrier = UOp.barrier(A_local_store, B_local_store)
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)

  # open inner k range
  k = UOp.range(BLOCK_K, 3, AxisType.REDUCE)

  # ---------------------------
  # LOCAL -> REG (per-wave tiles)
  # ---------------------------
  warp, lane = tid // WARP_SIZE, tid % WARP_SIZE
  waveIdx, waveIdy = warp % WAVES_PER_BLOCK_N, warp // WAVES_PER_BLOCK_N
  laneIdx, laneIdy = lane % LANES_PER_WAVE_N, lane // LANES_PER_WAVE_N
  assert waveIdy.vmax+1 == WAVES_PER_BLOCK_M and laneIdy.vmax+1 == LANES_PER_WAVE_M

  A_col = UOp.placeholder((REG_TILES_PER_WAVE_M, TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  A_local_slice = A_local[k, :].reshape(WAVES_PER_BLOCK_M, REG_TILES_PER_WAVE_M, LANES_PER_WAVE_M, TM)[waveIdy, :, laneIdy, :]
  A_col = A_col.after(copy(A_col, A_local_slice, 300, upcast=True))

  B_row = UOp.placeholder((REG_TILES_PER_WAVE_N, TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  B_local_slice = B_local[k, :].reshape(WAVES_PER_BLOCK_N, REG_TILES_PER_WAVE_N, LANES_PER_WAVE_N, TN)[waveIdx, :, laneIdx, :]
  B_row = B_row.after(copy(B_row, B_local_slice, 400, upcast=True))

  # ---------------------------
  # FMA: c_regs += A_col * B_row
  # ---------------------------
  c_regs = UOp.placeholder((REG_TILES_PER_WAVE_M, TM, REG_TILES_PER_WAVE_N, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  i = UOp.range(c_regs.size, 16)
  c_regs = c_regs.after(c_regs.flatten()[i].store(0.0).end(i))

  # TODO: why don't these work as upcast?
  # why if the ranges merge is it slow?!? (if you change the order on end, they will merge. big slowdown on METAL)
  iter_m, t_m, iter_n, t_n = rngs = rngs_for_shape(c_regs.shape, 500)
  sink = c_regs[*rngs].store(c_regs.after(k)[*rngs] + A_col[iter_m, t_m] * B_row[iter_n, t_n]).end(iter_m, iter_n, t_m, t_n)

  # Close k, sync, and close K tiles
  sink = sink.end(k).barrier().end(k_tile_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  c = c.reshape(WAVES_PER_BLOCK_M, REG_TILES_PER_WAVE_M, LANES_PER_WAVE_M, TM,
                WAVES_PER_BLOCK_N, REG_TILES_PER_WAVE_N, LANES_PER_WAVE_N, TN)
  c = c[waveIdy, :, laneIdy, :,
        waveIdx, :, laneIdx, :]
  sink = copy(c, c_regs.after(sink), rng=600)

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

def eval_custom_matmul(fxn, dt=dtypes.float):
  a = Tensor.randn(M, K, dtype=dt)
  b = Tensor.randn(K, N, dtype=dt)
  c = Tensor.empty(M, N, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(a, b)

  ets = []
  with Context(DEBUG=max(2, DEBUG.value)):
    for _ in range(NUM_RUNS):
      GlobalCounters.reset()
      tst = Tensor.custom_kernel(c, a, b, fxn=fxn)[0].realize()
      ets.append(GlobalCounters.time_sum_s)
  print(f"REAL TFLOPS {M * N * K * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a.float() @ b.float()).realize()
    with Context(DEBUG=0):
      err = (tc - tst).square().mean().item()
    print(f"mean squared error {err}")
    if err > (1e-2 if dt == dtypes.half else 1e-6):
      raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  eval_custom_matmul(hand_spec_kernel3)
