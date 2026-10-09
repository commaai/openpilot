import numpy as np
import halide as hl
from tinygrad.helpers import Timing, getenv

# HL_DEBUG_CODEGEN=1
N = getenv("N", 1024)

def gemm_pipeline(gpu=False):
  # ---------------- Vars & Parameters ----------------
  i, j = hl.Var("i"), hl.Var("j")  # output tile coordinates

  A = hl.InputBuffer(hl.Float(32), 2)  # [M, K]
  B = hl.InputBuffer(hl.Float(32), 2)  # [K, N]

  A.dim(0).set_bounds(0, N)
  A.dim(1).set_bounds(0, N)
  B.dim(0).set_bounds(0, N)
  B.dim(1).set_bounds(0, N)

  # ---------------- Definition ----------------

  k = hl.RDom([(0, N)])

  partial = hl.Func("partial")
  partial[i, j] = 0.0
  partial[i, j] += A[i, k] * B[k, j]

  C = hl.Func("C")
  C[i, j] = partial[i, j]

  if not gpu:
    # ---------------- Schedule ----------------
    VEC = 16
    TILE_I = 64
    TILE_J = 64

    io, jo, ii, ji = hl.Var("io"), hl.Var("jo"), hl.Var("ii"), hl.Var("ji")
    C.update().tile(i, j, io, jo, ii, ji, TILE_I, TILE_J).fuse(io, jo, io).parallel(io).vectorize(ji, VEC)
  else:
    # ---------------- Schedule ----------------
    GRP_I    = 8     # output tile size
    GRP_J    = 16

    #partial.store_in(hl.MemoryType.Register)
    #partial.update().unroll(k, 4)

    io, jo, ii, ji = hl.Var(), hl.Var(), hl.Var(), hl.Var()
    C.gpu_tile(i, j, io, jo, ii, ji, GRP_I, GRP_J, hl.TailStrategy.RoundUp)

  return C, A, B

if __name__ == "__main__":
  pipe, A, B = gemm_pipeline(gpu=True)

  # NOTE: meteal does nothing
  target = hl.get_host_target().with_feature(hl.TargetFeature.Metal)

  a_np = np.random.randn(N, N).astype(np.float32)
  b_np = np.random.randn(N, N).astype(np.float32)

  # reverse order is correct!
  a_hal = hl.Buffer(b_np)
  b_hal = hl.Buffer(a_np)
  A.set(a_hal)
  B.set(b_hal)

  pipe.compile_to_lowered_stmt("/tmp/my_function.html", [A, B], hl.StmtOutputFormat.HTML, target=target)
  #exit(0)

  c_hal = hl.Buffer(hl.Float(32), [N,N])
  with Timing("halide gemm "):
    pipe.realize(c_hal, target)
    c_hal.copy_to_host()
    c_out = np.array(c_hal)
  print(c_out)

  # tinygrad gets 60 ms with no BEAM, 20 ms with BEAM on CPU
  with Timing("halide gemm "):
    pipe.realize(c_hal, target)
    c_hal.copy_to_host()

  # Check correctness
  with Timing("numpy gemm "):
    ref = a_np @ b_np
  max_err = np.abs(ref - c_out).max()
  print("Max absolute error:", max_err)
  assert max_err < 1e-4, "GEMM result incorrect!"

  print("Pipeline ran on", target)
  print("Success - GEMM Halide-Python output matches NumPy.")
