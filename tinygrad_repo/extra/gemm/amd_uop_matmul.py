import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo, sint, AxisType
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv

N = getenv("N", 4096)
M = K = N
run_count = getenv("CNT", 5)

# ---------------------------
# launch/config constants
# ---------------------------

WARP_SIZE = 32

# Threadblock tile sizes (block-level tile of C that a block computes)
BLOCK_N = 128   # columns of C (N-dim) per block
BLOCK_M = 128   # rows of C (M-dim) per block
BLOCK_K = 8     # K-slice per block iteration

# Register tile sizes (per-thread accumulator tile of C)
TN = 4     # columns per thread
TM = 4     # rows per thread

is_kernel5 = getenv("K5", 0)
THREADS_PER_BLOCK = 128 if is_kernel5 else 256
assert THREADS_PER_BLOCK % BLOCK_N == 0, "THREADS_PER_BLOCK must be divisible by BLOCK_N"
assert THREADS_PER_BLOCK % BLOCK_K == 0, "THREADS_PER_BLOCK must be divisible by BLOCK_K"
assert (BLOCK_N * BLOCK_K) % THREADS_PER_BLOCK == 0
assert (BLOCK_M * BLOCK_K) % THREADS_PER_BLOCK == 0

WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE
WAVE_TILE_N = 128 if is_kernel5 else 64
WAVE_TILE_M = BLOCK_N * BLOCK_M // WARPS_PER_BLOCK // WAVE_TILE_N
assert BLOCK_N % WAVE_TILE_N == 0, "BN must be a multiple of WN"
assert BLOCK_M % WAVE_TILE_M == 0, "BM must be a multiple of WM"
WAVES_IN_BLOCK_X = BLOCK_N // WAVE_TILE_N
WAVES_IN_BLOCK_Y = BLOCK_M // WAVE_TILE_M
assert WAVES_IN_BLOCK_X * WAVES_IN_BLOCK_Y == WARPS_PER_BLOCK, "wave grid must match warps/block"

LANES_PER_WAVE_X = 8
LANES_PER_WAVE_Y = 4
ITERS_PER_WAVE_N = WAVE_TILE_N // (LANES_PER_WAVE_X * TN)
ITERS_PER_WAVE_M = WAVE_TILE_M // (LANES_PER_WAVE_Y * TM)
assert WAVE_TILE_N % (LANES_PER_WAVE_X * TN) == 0, "WAVE_TILE_N must be divisible by LANES_PER_WAVE_X*TN"
assert WAVE_TILE_M % (LANES_PER_WAVE_Y * TM) == 0, "WAVE_TILE_M must be divisible by LANES_PER_WAVE_Y*TM"

def rngs_for_shape(shape:tuple[sint, ...], rng:int, axis_type=AxisType.LOOP): return [UOp.range(s, rng+i, axis_type) for i,s in enumerate(shape)]
def copy(dest:UOp, src:UOp, rng:int, set=False, upcast=False):
  assert dest.shape == src.shape
  rngs = rngs_for_shape(src.shape, rng, AxisType.UPCAST if upcast else AxisType.LOOP)
  copy = dest[*rngs].store(src[*rngs]).end(*rngs)
  return dest.after(copy) if set else copy

def hand_spec_kernel3():
  # ---------------------------
  # block indices & placeholders
  # ---------------------------
  blockIdx_x = UOp.special(N // BLOCK_N, "gidx0")
  blockIdx_y = UOp.special(N // BLOCK_M, "gidx1")

  a = UOp.placeholder((N, N), dtypes.float, slot=1)
  b = UOp.placeholder((N, N), dtypes.float, slot=2)
  c = UOp.placeholder((N, N), dtypes.float, slot=0)

  # index the output with the globals
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[blockIdx_y, :, blockIdx_x, :]

  # open the main reduction range
  k_tile_range = UOp.range(N // BLOCK_K, 0, AxisType.REDUCE)
  a = a.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_K, BLOCK_K)[blockIdx_y, :, k_tile_range, :]
  b = b.reshape(N // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N)[k_tile_range, :, blockIdx_x, :]

  # globals are no longer used, they are already in the indexes
  del blockIdx_y, blockIdx_x

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

  # A: read BM x BK tiles (permute on store into locals)
  BM_As_stride = (BLOCK_M + 4) if is_kernel5 else BLOCK_M
  As = UOp.placeholder((BLOCK_K, BM_As_stride), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_M))
  As_store = copy(As.permute((1,0)).reshape(-1, THREADS_PER_BLOCK)[:, tid], a.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=100)

  # B: read BK x BN tiles
  Bs = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)
  Bs_store = copy(Bs.reshape(-1, THREADS_PER_BLOCK)[:, tid], b.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=200)

  # TODO: can we automate barrier?
  barrier = UOp.barrier(As_store, Bs_store)
  As, Bs = As.after(barrier), Bs.after(barrier)

  # open inner k range
  k = UOp.range(BLOCK_K, 3, AxisType.REDUCE)

  # ---------------------------
  # LOCAL -> REG (per-wave tiles)
  # ---------------------------
  waveIdx = (tid // WARP_SIZE) % WAVES_IN_BLOCK_X
  waveIdy = (tid // WARP_SIZE) // WAVES_IN_BLOCK_X
  assert waveIdy.vmax+1 == WAVES_IN_BLOCK_Y

  laneIdx = (tid % WARP_SIZE) % LANES_PER_WAVE_X
  laneIdy = (tid % WARP_SIZE) // LANES_PER_WAVE_X
  assert laneIdy.vmax+1 == LANES_PER_WAVE_Y

  A_col = UOp.placeholder((ITERS_PER_WAVE_M, TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  A_col = copy(A_col, As[k, :].reshape(WAVES_IN_BLOCK_Y, ITERS_PER_WAVE_M, LANES_PER_WAVE_Y, TM)[waveIdy, :, laneIdy, :], 300, set=True, upcast=True)

  B_row = UOp.placeholder((ITERS_PER_WAVE_N, TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  B_row = copy(B_row, Bs[k, :].reshape(WAVES_IN_BLOCK_X, ITERS_PER_WAVE_N, LANES_PER_WAVE_X, TN)[waveIdx, :, laneIdx, :], 400, set=True, upcast=True)

  # ---------------------------
  # FMA: c_regs += A_col * B_row
  # ---------------------------
  c_regs = UOp.placeholder((ITERS_PER_WAVE_M, TM, ITERS_PER_WAVE_N, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  i = UOp.range(c_regs.size, 16)
  c_regs = c_regs.after(c_regs.flatten()[i].store(0.0).end(i))

  # TODO: why don't these work as upcast?
  # why if the ranges merge is it slow?!? (if you change the order on end, they will merge. big slowdown on METAL)
  iterWaveM, yt, iterWaveN, xt = rngs = rngs_for_shape(c_regs.shape, 500)
  sink = c_regs[*rngs].store(c_regs.after(k)[*rngs] + A_col[iterWaveM, yt] * B_row[iterWaveN, xt]).end(iterWaveM, iterWaveN, yt, xt)

  # Close k, sync, and close K tiles
  sink = sink.end(k).barrier().end(k_tile_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  c = c.reshape(WAVES_IN_BLOCK_Y, ITERS_PER_WAVE_M, LANES_PER_WAVE_Y, TM,
                WAVES_IN_BLOCK_X, ITERS_PER_WAVE_N, LANES_PER_WAVE_X, TN)
  c = c[waveIdy, :, laneIdy, :,
        waveIdx, :, laneIdx, :]
  sink = copy(c, c_regs.after(sink), rng=600)

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

def test_matmul(sink:UOp, dtype=dtypes.float32, N=N):
  rng = np.random.default_rng()
  a = Tensor(rng.random((N, N), dtype=np.float32)-0.5, dtype=dtype)
  b = Tensor(rng.random((N, N), dtype=np.float32)-0.5, dtype=dtype)
  hc = Tensor.empty(N, N, dtype=dtype)
  Tensor.realize(a, b, hc)

  ei = ExecItem(sink, [t.uop.buffer for t in [hc, a, b]], prg=get_runner(Device.DEFAULT, sink))

  ets = []
  with Context(DEBUG=2):
    for _ in range(run_count):
      ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a @ b).realize()
    with Context(DEBUG=0):
      err = (hc - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-06:
      raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  test_matmul(hand_spec_kernel3(), N=N)
