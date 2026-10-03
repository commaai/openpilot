"""
tilelang-style matmul_relu written with tinygrad UOp APIs.

Reference tilelang kernel:
  @tilelang.jit
  def matmul_relu(A, B, block_M=64, block_N=64, block_K=64,
                  dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const('M, N, K')
    C = T.empty([M, N], dtype)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
      A_shared = T.alloc_shared((block_M, block_K), dtype)
      B_shared = T.alloc_shared((block_K, block_N), dtype)
      C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
      T.clear(C_local)
      for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
      for i, j in T.Parallel(block_M, block_N):
        C_local[i, j] = T.max(C_local[i, j], 0)
      T.copy(C_local, C[by * block_M, bx * block_N])
    return C
"""

from tinygrad.dtype import dtypes, AddrSpace, DType
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.helpers import cdiv, getenv
from tinygrad.tensor import Tensor

# ---------------------------------------------------------------------------
# tilelang builtins, expressed with tinygrad UOp APIs
# ---------------------------------------------------------------------------

def alloc_shared(shape:tuple[int, ...], dtype:DType, slot:int) -> UOp:
  """T.alloc_shared: one LOCAL buffer shared by all threads in the block."""
  return UOp.placeholder(tuple(shape), dtype, slot, AddrSpace.LOCAL)

def alloc_fragment(shape:tuple[int, ...], dtype:DType, slot:int, axes:tuple[int, ...], rngs:tuple[UOp, ...]) -> UOp:
  """T.alloc_fragment: per-thread REG fragment + UNSHARD over the LOCAL thread grid."""
  assert len(axes) == len(rngs)
  assert all(tnum.op is Ops.RANGE and tnum.arg[-1] is AxisType.LOCAL for tnum in rngs), "fragments shard over LOCAL ranges"
  by_axis = dict(zip(axes, rngs))
  shard_shape = tuple(s // (int(by_axis[i].vmax)+1) if i in by_axis else s for i, s in enumerate(shape))
  fragment = UOp.placeholder(shard_shape, dtype, slot, AddrSpace.REG)
  return fragment.unshard(axes, rngs)

# ---------------------------------------------------------------------------
# GEMM kernel: C = relu(A @ B), float inputs (fp16 or fp32), fp32 fragment accumulator, no WMMA
# ---------------------------------------------------------------------------

# 64x64 output tile per block, 128 threads as an 8x16 grid; each thread owns an 8x4 fragment sub-tile
# (the 2-D per-thread layout tilelang infers for this GEMM). The 4 contiguous columns (TN=4) are what
# let codegen vectorize loads/stores to float4, matching tilelang's lowering exactly.
BLOCK_M = BLOCK_N = BLOCK_K = 64
TY = 8
TX = 16
THREADS = TY * TX
TM = BLOCK_M // TY   # fragment rows per thread  (8)
TN = BLOCK_N // TX   # fragment columns per thread (4)

def matmul_relu_kernel(c:UOp, a:UOp, b:UOp) -> UOp:
  """C[M, N] = relu(A[M, K] @ B[K, N]) -- one 64x64 tile per block, locals + a 2-D fragment."""
  M, K = a.shape
  K2, N = b.shape
  assert K == K2 and a.dtype == b.dtype == c.dtype and not dtypes.is_int(a.dtype)
  assert not (K % BLOCK_K or M % BLOCK_M or N % BLOCK_N), "test sizes must be multiples of the block sizes"

  # with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
  bx = UOp.range(cdiv(N, BLOCK_N), 0, AxisType.GLOBAL)
  by = UOp.range(cdiv(M, BLOCK_M), 1, AxisType.GLOBAL)

  # 16*8 threads = 128 threads
  tx = UOp.range(TX, 2, AxisType.LOCAL)
  ty = UOp.range(TY, 3, AxisType.LOCAL)

  # shared + fragment (regs)
  A_shared = alloc_shared((BLOCK_M, BLOCK_K), a.dtype, 0)
  B_shared = alloc_shared((BLOCK_K, BLOCK_N), b.dtype, 1)
  C_local = alloc_fragment((TM, TY, TX, TN), dtypes.float32, 0, (1, 2), (ty, tx))

  # zero out the regs to start. this is expanded by the devectorizer
  C_local = C_local.after(C_local.store(0.0))

  # for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
  ko = UOp.range(cdiv(K, BLOCK_K), 6, AxisType.LOOP)

  # index the outer matrices
  a = a.rearrange("(m bm) (k bk) -> m k bm bk", bm=BLOCK_M, bk=BLOCK_K)[by, ko]
  b = b.rearrange("(k bk) (n bn) -> k n bk bn", bk=BLOCK_K, bn=BLOCK_N)[ko, bx]
  c = c.rearrange("(m bm) (n bn) -> m n bm bn", bm=BLOCK_M, bn=BLOCK_N)[by, bx]

  # T.copy: A_shared <- a, B_shared <- b
  def with_threads(x:UOp): return x.rearrange("(tm ty) (tx tn) -> ty tx tm tn", tm=TM, tn=TN)[ty, tx]
  A_shared = A_shared.after(with_threads(A_shared).store(with_threads(a)))
  B_shared = B_shared.after(with_threads(B_shared).store(with_threads(b)))

  # T.gemm(A_shared, B_shared, C_local), no WMMA
  kk = UOp.range(BLOCK_K, 11, AxisType.LOOP)
  ir = UOp.range(TM, 12, AxisType.LOOP)
  jj = UOp.range(TN, 13, AxisType.UPCAST)
  acc = C_local.after(kk)[ir, ty, tx, jj] + A_shared[ir*TM + ty, kk].cast(dtypes.float32) * B_shared[kk, tx*TN + jj].cast(dtypes.float32)
  # closing the ko loop here too; codegen adds the barrier so no thread overwrites the tiles while others still read them
  C_local = C_local[ir, ty, tx, jj].set(acc, end=(kk, ir, jj, ko))

  # c <- C_local (with relu and cast): every thread stores its shard's sub-view of the output tile
  c_st = c.reshape(C_local.shape).store(C_local.relu().cast(c.dtype))

  # close the locals and globals
  return c_st.end(tx, ty, bx, by).sink(arg=KernelInfo(name="matmul_relu", opts_to_apply=()))

# ---------------------------------------------------------------------------
# python wrapper: same signature as the tilelang function
# ---------------------------------------------------------------------------

def matmul_relu(a:Tensor, b:Tensor) -> Tensor:
  """C = relu(A @ B), fp16 in/out with an fp32 fragment accumulator."""
  c = Tensor.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
  return c.custom_kernel(a, b, fxn=matmul_relu_kernel)[0]

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  from tinygrad import Device
  assert Device[Device.DEFAULT].renderer.has_local, "this GPU-style kernel needs a backend with local memory (LOCAL ranges + barriers)"
  M = K = N = getenv("N", 256)  # 4x4 grid of 64x64 tiles, 4 K chunks
  dtype_in = dtypes.half if getenv("HALF") else dtypes.float

  a = Tensor.randn(M, K, dtype=dtype_in).contiguous()
  b = Tensor.randn(K, N, dtype=dtype_in).contiguous()
  ref = (a @ b).relu().realize()

  for _ in range(10):
    out = matmul_relu(a, b).realize()

  import numpy as np
  np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-1, rtol=1e-2)
  print("matmul_relu passed!")
