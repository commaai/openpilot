from tinygrad import UOp, dtypes
from tinygrad.uop.ops import AxisType, Ops, KernelInfo, AddrSpace
from extra.gemm.amd_uop_matmul import test_matmul

N = 2048

# metal has an 8x8 tensor core. this is the indexing
def mat_idx(buf, g0, g1, warp, u):
  l = [(warp//2**i)%2 for i in range(5)]
  return buf[g0, l[4]*4 + l[2]*2 + l[1], g1, l[3]*4 + l[0]*2 + u]

def hand_spec_tc_cores():
  gx = UOp.special(N // 8, "gidx0")
  gy = UOp.special(N // 8, "gidx1")
  warp = UOp.special(32, "lidx0")

  c = UOp.placeholder((N, N), dtypes.float, slot=0).reshape((N//8, 8, N//8, 8))
  a = UOp.placeholder((N, N), dtypes.float, slot=1).reshape((N//8, 8, N//8, 8))
  b = UOp.placeholder((N, N), dtypes.float, slot=2).reshape((N//8, 8, N//8, 8))

  gk = UOp.range(N // 8, 0, AxisType.REDUCE)

  a_tc = UOp.vectorize(*[mat_idx(a, gx, gk, warp, i) for i in range(2)])
  b_tc = UOp.vectorize(*[mat_idx(b, gk, gy, warp, i) for i in range(2)])

  acc = UOp.placeholder((2,), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  acc = acc[0].set(0.0)
  acc = acc[1].set(0.0)

  # TODO: make this simple
  wmma_arg = ('WMMA_8_8_8_float_float', (8, 8, 8), dtypes.float, dtypes.float, 'METAL', 32, (((3, 2),), ((3, 2),), ((3, 2),)), ())

  acc_load = UOp.vectorize(acc.after(gk)[0], acc.after(gk)[1])
  out = UOp(Ops.WMMA, dtypes.float.vec(2), (a_tc, b_tc, acc_load), arg=wmma_arg)

  end_loop = UOp.group(*[acc[i].store(out.gep(i)) for i in range(2)]).end(gk)

  sink = UOp.group(*[mat_idx(c.after(end_loop), gx, gy, warp, i).store(acc[i]) for i in range(2)])
  return sink.sink(arg=KernelInfo(name="custom_metal_matmul", opts_to_apply=())).simplify()

if __name__ == "__main__":
  test_matmul(hand_spec_tc_cores(), N=N)
