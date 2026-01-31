import os
import numpy as np
np.set_printoptions(linewidth=1000000)
os.environ["AMD_LLVM"] = "0"

from tinygrad import Tensor, Context, dtypes, UOp, GlobalCounters
from tinygrad.helpers import DEBUG, getenv
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import sint, AxisType, KernelInfo, Ops

WARP_SIZE = 64

# Reg tile sizes (tensor cores)
TC_M = 16
TC_N = 16
TC_K = 32

N,M,K = 4096,4096,4096

# Threadblock tile sizes (block-level tile of C that a block computes)
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 64

WARPGROUP_SIZE = 1
BLOCK_M = BLOCK_M * WARPGROUP_SIZE

TID_SIZE = WARPGROUP_SIZE*WARP_SIZE

def copy(dest:UOp, src:UOp, rng:int, set=False, upcast=()):
  assert dest.shape == src.shape
  rngs = [UOp.range(s, rng+i, AxisType.UPCAST if i in upcast else AxisType.LOOP) for i,s in enumerate(src.shape)]
  copy = dest[*rngs].store(src[*rngs]).end(*rngs)
  return dest.after(copy) if set else copy

def compute_on_locals(acc:UOp, Asl:UOp, Bsl:UOp, rng:int, afters:tuple[UOp, ...], warpgroup, warp) -> UOp:
  K_inner_loop = UOp.range(BLOCK_K//TC_K, rng, AxisType.REDUCE)

  # load from locals into registers
  Ar = UOp.placeholder((BLOCK_M//TC_M//WARPGROUP_SIZE,), dtypes.half.vec(8), slot=1, addrspace=AddrSpace.REG)
  Br = UOp.placeholder((BLOCK_N//TC_N,), dtypes.half.vec(8), slot=2, addrspace=AddrSpace.REG)

  M_load_loop = UOp.range(BLOCK_M//TC_M//WARPGROUP_SIZE, rng+10)
  Asl = Asl.reshape(BLOCK_K//TC_K, TC_K, BLOCK_M//TC_M//WARPGROUP_SIZE, WARPGROUP_SIZE, TC_M)
  load_rng = UOp.range(8, rng+11, axis_type=AxisType.UPCAST)
  A_in = Asl[K_inner_loop, (warp//16)*8+load_rng, M_load_loop, warpgroup, warp%16].contract(load_rng)
  Ar = Ar[M_load_loop].set(A_in, end=M_load_loop)

  N_load_loop = UOp.range(BLOCK_N//TC_N, rng+20)
  Bsl = Bsl.reshape(BLOCK_K//TC_K, TC_K, BLOCK_N//TC_N, TC_N)
  load_rng = UOp.range(8, rng+21, axis_type=AxisType.UPCAST)
  B_in = Bsl[K_inner_loop, (warp//16)*8+load_rng, N_load_loop, warp%16].contract(load_rng)
  Br = Br[N_load_loop].set(B_in, end=N_load_loop)

  M_inner_loop = UOp.range(BLOCK_M//TC_M//WARPGROUP_SIZE, rng+30)
  N_inner_loop = UOp.range(BLOCK_N//TC_N, rng+31)

  # load values
  acc_after = acc.after(*afters, M_inner_loop, N_inner_loop, K_inner_loop)
  acc_load = acc_after[N_inner_loop, M_inner_loop]

  # do WMMA
  wmma_arg = ('WMMA_16_16_32_half_float', (16, 16, 32), dtypes.half, dtypes.float, 'AMD', 64, ((), (), ((3, 2), (2, 2))), ())
  out = UOp(Ops.WMMA, dtypes.float.vec(4), (Ar[M_inner_loop], Br[N_inner_loop], acc_load), arg=wmma_arg)

  # store back the acc
  acc_store = acc[N_inner_loop, M_inner_loop].store(out)
  return acc_store.end(M_inner_loop, N_inner_loop, K_inner_loop)

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  gx, gy = UOp.special(M//BLOCK_M, "gidx0"), UOp.special(N//BLOCK_N, "gidx1")
  K_outer_loop = UOp.range(K//BLOCK_K, 0, AxisType.REDUCE)

  # split out the globals into blocks
  C = C.src[0].cast(dtypes.float.vec(4).ptr(C.ptrdtype.size)).reshape((M//BLOCK_M, BLOCK_M, N//BLOCK_N, BLOCK_N))
  A = A.reshape((M//BLOCK_M, BLOCK_M, K//BLOCK_K, BLOCK_K))[gx, :, K_outer_loop, :]
  B = B.reshape((K//BLOCK_K, BLOCK_K, N//BLOCK_N, BLOCK_N))[K_outer_loop, :, gy, :]

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  tid = UOp.special(TID_SIZE, "lidx0")
  warpgroup, warp = tid//WARP_SIZE, tid%WARP_SIZE

  A_view = A.reshape(-1, TID_SIZE, 8)
  B_view = B.reshape(-1, TID_SIZE, 8)

  # A: read BM x BK tiles (permute on store into locals)
  As = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL).shrink_to(BLOCK_K, BLOCK_M)
  As_view = As.reshape(-1, TID_SIZE, 8)

  Bs = UOp.placeholder((BLOCK_K, BLOCK_N+4), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL).shrink_to(BLOCK_K, BLOCK_N)
  Bs_view = Bs.reshape(-1, TID_SIZE, 8)

  outer_copy = UOp.range(A_view.shape[0], 100, AxisType.UPCAST)
  inner_copy = UOp.range(A_view.shape[2], 101, AxisType.UPCAST)
  As_store = As_view[outer_copy, tid, inner_copy].store(A_view[outer_copy, tid, inner_copy])
  Bs_store = Bs_view[outer_copy, tid, inner_copy].store(B_view[outer_copy, tid, inner_copy])

  if getenv("NOLOAD"):
    As_store = As[0,0].store(0)
    Bs_store = Bs[0,0].store(0)

  # TODO: can we automate barrier?
  barrier = UOp.barrier(UOp.group(As_store, Bs_store).end(outer_copy, inner_copy))

  if getenv("COMPUTE"):
    As, Bs = As.after(barrier), Bs.after(barrier)

    acc = UOp.placeholder((BLOCK_N//TC_N, BLOCK_M//TC_M//WARPGROUP_SIZE), dtypes.float.vec(4), 0, AddrSpace.REG)

    sink = compute_on_locals(acc, As, Bs, 200, afters=(barrier,), warpgroup=warpgroup, warp=warp)
    sink = sink.end(K_outer_loop)

    C_view = C[gx, :, gy, :].reshape(BLOCK_M//TC_M//WARPGROUP_SIZE, WARPGROUP_SIZE, TC_M, BLOCK_N//TC_N, TC_N)[:, warpgroup, warp%16, :, (warp//16)*4]
    sink = copy(C_view, acc.after(sink), rng=300)
  else:
    sink = C.after(barrier.end(K_outer_loop))[0,0,0,0].store(As[0,0]+Bs[0,0])

  return sink.sink(arg=KernelInfo(name="custom_gemm", opts_to_apply=())).simplify()

if __name__ == "__main__":
  a = Tensor.randn(M, K, dtype=dtypes.half)
  b = Tensor.randn(K, N, dtype=dtypes.half)
  c = Tensor.empty(M, N, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(a,b)


  GlobalCounters.reset()
  with Context(DEBUG=max(2, DEBUG.value), DEVECTORIZE=2):
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    tst.realize()
  print(f"{(N*M*K*2 / GlobalCounters.time_sum_s)*1e-12:.2f} REAL TFLOPS")


  with Context(DEBUG=0):
    ref = a.dot(b, dtype=dtypes.float)
    ref.realize()
    #print(ref.numpy())
    #print(tst.numpy())
    assert Tensor.isclose(ref, tst, atol=1e-2).all().item(), "matrix not close"
