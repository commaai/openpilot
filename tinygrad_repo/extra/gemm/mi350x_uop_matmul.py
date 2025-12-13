import os
import numpy as np
np.set_printoptions(linewidth=1000000)
os.environ["AMD_LLVM"] = "0"

from tinygrad import Tensor, Context, dtypes, UOp, GlobalCounters
from tinygrad.helpers import DEBUG, getenv
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

WARP_SIZE = 64

# Reg tile sizes (tensor cores)
TC_M = 16
TC_N = 16
TC_K = 32

# 1024 matrix cores
# 16 cycle mfma
# 2.2 GHz
# 16x16x32x2 FLOPS/mma = 16384
# 2.2*1e9*16384*1024/16*1e-12 TFLOPS = 2306 TFLOPS

#N,M,K = 256,256,64
N,M,K = 4096,4096,4096

# Threadblock tile sizes (block-level tile of C that a block computes)
#BLOCK_M = 128   # rows of C (M-dim) per block
#BLOCK_N = 128   # columns of C (N-dim) per block
#BLOCK_K = 128   # K-slice per block iteration

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 128

WARPGROUP_SIZE = 1
BLOCK_M = BLOCK_M * WARPGROUP_SIZE

# TODO: improve the syntax of this. better syntax, faster iteration
#  -- DONE: add working slice a[gx, :, i] -> shape of the : (aka (16,16,32) becomes (16,))
#  -- DONE(ish): add argfix to movement (traits shared with Tensor)
#  -- fix WMMA to not require all the junk
#  -- improve syntax for vectorized loads/stores (both with DEVECTORIZE and without)
#  -- DONE: be able to use CONTRACT on a range
#  -- fix upcasted RANGE on an already vectorized buffer
#  -- improve "all ranges not ended error" / fix the bug with after on ended ranges (if you are after end of range, range is closed)

CUS_PER_GPU = 256
assert ((M//BLOCK_M) * (N//BLOCK_N)) >= CUS_PER_GPU, "not enough globals"

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  # A = (M x K)
  # B = (K x N)
  # C = (M x N)

  # check it's proper matmul
  assert C.shape[0] == A.shape[0]
  assert C.shape[1] == B.shape[1]
  assert A.shape[1] == B.shape[0]

  gx, gy = UOp.special(M//BLOCK_M, "gidx0"), UOp.special(N//BLOCK_N, "gidx1")
  warp = UOp.special(WARP_SIZE, "lidx0")
  warpgroup = UOp.special(WARPGROUP_SIZE, "lidx1")

  # generic copy logic (not good)
  def generic_copy(glbl, gargs, lcl, rng):
    # Fully coalesced 128-bit loads/stores.
    INNER_SIZE = 8
    cp_i = UOp.range(lcl.size//(WARPGROUP_SIZE*WARP_SIZE*INNER_SIZE), rng)
    cp_inner = UOp.range(INNER_SIZE, rng+1, AxisType.UPCAST)
    idx_i = cp_i*WARPGROUP_SIZE*WARP_SIZE*INNER_SIZE + warpgroup*WARP_SIZE*INNER_SIZE + warp*INNER_SIZE + cp_inner
    return lcl[idx_i].store(glbl[*gargs, idx_i]).end(cp_i, cp_inner)

  # split out the globals into blocks
  C = C.reshape((M//BLOCK_M, BLOCK_M, N//BLOCK_N, BLOCK_N))
  A = A.reshape((M//BLOCK_M, BLOCK_M, K//BLOCK_K, BLOCK_K))
  B = B.reshape((K//BLOCK_K, BLOCK_K, N//BLOCK_N, BLOCK_N))

  # this is the big accumulator
  acc = UOp.placeholder((BLOCK_N//TC_N, BLOCK_M//TC_M//WARPGROUP_SIZE), dtypes.float.vec(4), 0, AddrSpace.REG)
  assert acc.size*WARP_SIZE*WARPGROUP_SIZE*4 == BLOCK_M*BLOCK_N
  acc = acc[init_l:=UOp.range(acc.size, 500)].set(UOp.const(dtypes.float.vec(4), 0.0), end=init_l)

  # create locals (note A is permuted, and the stride is changed to avoid bank conflicts)
  def make_locals(slot) -> tuple[UOp, UOp]:
    BM_As_stride = (BLOCK_M + 1)
    BN_Bs_stride = (BLOCK_N + 0)
    INNER_SLICE = 8
    As = UOp.placeholder((BLOCK_K//INNER_SLICE, BM_As_stride, INNER_SLICE), dtypes.half, slot=slot, addrspace=AddrSpace.LOCAL)
    INNER_SLICE = 1
    Bs = UOp.placeholder((BLOCK_K//INNER_SLICE, BN_Bs_stride, INNER_SLICE), dtypes.half, slot=slot+1, addrspace=AddrSpace.LOCAL)
    As = As.permute((0,2,1)).reshape((BLOCK_K, BM_As_stride)).shrink_to((BLOCK_K, BLOCK_M))
    Bs = Bs.permute((0,2,1)).reshape((BLOCK_K, BN_Bs_stride)).shrink_to((BLOCK_K, BLOCK_N))
    return As, Bs

  # load from globals into locals (TODO: use the warpgroup)

  def load_to_locals(l_K_outer_loop:UOp, Asl:UOp, Bsl:UOp, rng:int, barrier=True) -> tuple[UOp, UOp]:
    if getenv("FAKE"):
      return Asl[0].set(0), Bsl[0].set(0)
    else:
      pA = A.permute((0,2,1,3)).reshape((M//BLOCK_M, K//BLOCK_K, BLOCK_M*BLOCK_K))
      pas = Asl.permute((1,0)).reshape((BLOCK_M*BLOCK_K,))
      As_store = generic_copy(pA, (gx, l_K_outer_loop), pas, rng)

      pB = B.permute((0,2,1,3)).reshape((K//BLOCK_K, N//BLOCK_N, BLOCK_K*BLOCK_N))
      pbs = Bsl.reshape((BLOCK_K*BLOCK_N,))
      Bs_store = generic_copy(pB, (l_K_outer_loop, gy), pbs, rng+2)

      barrier = UOp.barrier(As_store, Bs_store) if barrier else UOp.group(As_store, Bs_store)
      return Asl.after(barrier), Bsl.after(barrier)

  def compute_on_locals(acc:UOp, Asl:UOp, Bsl:UOp, rng:int, afters:tuple[UOp, ...]=()) -> UOp:
    K_inner_loop = UOp.range(BLOCK_K//TC_K, rng, AxisType.REDUCE)

    # load from locals into registers
    Ar = UOp.placeholder((BLOCK_M//TC_M//WARPGROUP_SIZE,), dtypes.half.vec(8), slot=1, addrspace=AddrSpace.REG)
    Br = UOp.placeholder((BLOCK_N//TC_N,), dtypes.half.vec(8), slot=2, addrspace=AddrSpace.REG)

    M_load_loop = UOp.range(BLOCK_M//TC_M//WARPGROUP_SIZE, rng+10)
    Asl = Asl.reshape((BLOCK_K//TC_K, TC_K, BLOCK_M//TC_M//WARPGROUP_SIZE, WARPGROUP_SIZE, TC_M))
    load_rng = UOp.range(8, rng+11, axis_type=AxisType.UPCAST)
    A_in = Asl[K_inner_loop, (warp//16)*8+load_rng, M_load_loop, warpgroup, warp%16].contract(load_rng)
    Ar = Ar[M_load_loop].set(A_in, end=M_load_loop)

    N_load_loop = UOp.range(BLOCK_N//TC_N, rng+20)
    Bsl = Bsl.reshape((BLOCK_K//TC_K, TC_K, BLOCK_N//TC_N, TC_N))
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

  # **** START INNER LOOP *****
  # inner loop -- locals -> regs

  # no pipeline
  if not getenv("PIPELINE"):
    As, Bs = make_locals(slot=0)

    K_outer_loop = UOp.range(K//BLOCK_K, 0, AxisType.REDUCE)
    As, Bs = load_to_locals(K_outer_loop, As, Bs, 1000, barrier=True)
    acc_store = compute_on_locals(acc, As, Bs, 1500, afters=(K_outer_loop,))
    acc = acc.after(acc_store.barrier().end(K_outer_loop))
  else:
    # this doesn't work
    As0, Bs0 = make_locals(slot=0)
    As1, Bs1 = make_locals(slot=2)
    As0, Bs0 = load_to_locals(0, As0, Bs0, 1000)

    K_outer_loop = UOp.range((K//BLOCK_K-2)//2, 0, AxisType.REDUCE)
    As1, Bs1 = load_to_locals(K_outer_loop+1, As1, Bs1, 2000, barrier=False)
    acc_store = compute_on_locals(acc, As0, Bs0, 1500, afters=(K_outer_loop,))
    As0, Bs0 = load_to_locals(K_outer_loop+2, As0, Bs0, 3000, barrier=False)
    acc_store = compute_on_locals(acc, As1, Bs1, 2500, afters=(acc_store, As0, Bs0))
    acc = acc.after(acc_store.barrier().end(K_outer_loop))

    #acc_store = compute_on_locals(acc, As0, Bs0, 3500, afters=(acc_store.barrier().end(K_outer_loop)))
    """
    As1, Bs1 = load_to_locals(K//BLOCK_K-1, As1, Bs1, 4000)
    acc_store = compute_on_locals(acc, As1, Bs1, 4500, afters=(acc_store))
    """
    #acc = acc.after(acc_store)

  # **** END LOOPS *****

  # store the acc into gmem
  cp_i, cp_j = UOp.range(BLOCK_M//TC_M//WARPGROUP_SIZE, 10004), UOp.range(BLOCK_N//TC_N, 10005)
  c_load = lambda i: C[gx, cp_i*TC_M*WARPGROUP_SIZE + warpgroup*TC_M + (warp//16)*4+i, gy, cp_j*TC_N + warp%16]
  store = UOp.group(*[c_load(i).store(acc[cp_j, cp_i].gep(i)) for i in range(4)])
  store = store.end(cp_i, cp_j)

  return store.sink(arg=KernelInfo(name="custom_gemm", opts_to_apply=())).simplify()

# simplest WMMA
"""
# init the acc
acc = UOp.placeholder((4,), dtypes.float, 0, AddrSpace.REG)
acc = acc[init_l:=UOp.range(4, 1)].set(0.0, end=init_l)

# do the wmma
acc_load = UOp.vectorize(*[acc.after(K_loop)[i] for i in range(4)])
wmma_arg = ('WMMA_16_16_32_half_float', (16, 16, 32), dtypes.half, dtypes.float, 'AMD', 64, ((), (), ((3, 2), (2, 2))), ())
out = UOp(Ops.WMMA, dtypes.float.vec(4), (A_in, B_in, acc_load), arg=wmma_arg)

# store back the acc
acc = acc.after(UOp.group(*[acc[i].store(out.gep(i)) for i in range(4)]).end(K_loop))

# store the acc into gmem
store = UOp.group(*[C[gx, (warp//16)*4+i, gy, warp%16].store(acc[i]) for i in range(4)])
"""

if __name__ == "__main__":
  a = Tensor.randn(M, K, dtype=dtypes.half)
  b = Tensor.randn(K, N, dtype=dtypes.half)

  #a = Tensor.zeros(M, K, dtype=dtypes.half).contiguous()
  #a[0,16] = 1
  #b = Tensor.ones(K, N, dtype=dtypes.half).contiguous()

  c = Tensor.empty(M, N, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(a,b)

  ref = a.dot(b, dtype=dtypes.float)
  ref.realize()

  GlobalCounters.reset()
  with Context(DEBUG=max(2, DEBUG.value), DEVECTORIZE=2):
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    tst.realize()
  print(f"{(N*M*K*2 / GlobalCounters.time_sum_s)*1e-12:.2f} REAL TFLOPS")

  with Context(DEBUG=0):
    #print(ref.numpy())
    #print(tst.numpy())
    assert Tensor.isclose(ref, tst, atol=1e-2).all().item(), "matrix not close"
