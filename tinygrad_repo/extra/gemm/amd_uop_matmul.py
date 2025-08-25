from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, graph_rewrite, AxisType, PatternMatcher, UPat
from tinygrad.engine.realize import CompiledRunner, ExecItem, get_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv, colored, prod, unwrap
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.shape.view import strides_for_shape
from tinygrad.codegen.opt.kernel import axis_colors
from tinygrad.codegen.opt.swizzler import merge_views, view_left

def to_colored(full_shape, axis_types): return '_'.join([colored(str(s), axis_colors[at]) for s,at in zip(full_shape, axis_types)])

N = 4096
run_count = 5

BN = 128
BM = 128
BK = 8

TN = 4
TM = 4

# NOTE: this is from testgrad
# change reduceop axes and input ShapeTrackers, view gets replaced with a reshape.
# src->r->view  -->   src->view->r
def swizzle_reduceop(src:UOp, r:UOp, view:UOp):
  if r.tag is not None: return None
  # confirm the input is in order
  # TODO: replace this with a UOp that allows for nothing else then remove this
  permute = tuple(i for i in range(len(src.shape)) if i not in r.axis_arg)+r.axis_arg
  assert permute == tuple(range(len(permute))), f"reduce axis must already be in order, {permute} isn't"

  # append the reduce shape to each of the views
  prshape = prod(rshape:=src.shape[-len(r.axis_arg):])
  rstrides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+rstrides, v.offset*prshape,
                    v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in unwrap(view.st).views]

  # no reshape required with shrinking REDUCE_AXIS
  return UOp(Ops.REDUCE_AXIS, r.dtype, (src.view(ShapeTracker(tuple(nv))),),
             (r.arg[0], tuple(range(len(view.shape), len(view.shape) + len(r.axis_arg)))))

pm = PatternMatcher([
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
])

def top_spec_kernel3():
  a = Tensor.empty(N,N)
  b = Tensor.empty(N,N)
  c = a@b
  sink = c.schedule()[-1].ast
  L = 16
  sink = sink.reshape((N//L, L, N//L, L)) #.lift({0:UOp.range(dtypes.int, N//BM, 0), 2:UOp.range(dtypes.int, N//BN, 1)})
  sink = graph_rewrite(sink, view_left+pm)
  axis_types = (AxisType.GLOBAL, AxisType.LOCAL, AxisType.GLOBAL, AxisType.LOCAL, AxisType.REDUCE)
  return sink.replace(arg=KernelInfo(name="top_"+to_colored(sink.full_shape, axis_types), axis_types=axis_types))

def hl_spec_kernel3():
  nbIterWaveM = 2
  nbIterWaveN = 2

  # define buffers
  # TODO: remove these views once the defines have a shape
  a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1).view(ShapeTracker.from_shape((N,N)))
  b = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2).view(ShapeTracker.from_shape((N,N))).permute((1,0))
  c = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0).view(ShapeTracker.from_shape((N,N)))
  As = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BM, AddrSpace.LOCAL), arg=0).view(ShapeTracker.from_shape((BK, BM))).permute((1,0))
  Bs = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BN, AddrSpace.LOCAL), arg=1).view(ShapeTracker.from_shape((BK, BN))).permute((1,0))
  A_col = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveM * TM, AddrSpace.REG), arg=0).view(ShapeTracker.from_shape((nbIterWaveM * TM,)))
  B_row = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveN * TN, AddrSpace.REG), arg=1).view(ShapeTracker.from_shape((nbIterWaveN * TN,)))

  # shape buffers. TODO: permutes
  full_shape = (N//BM, nbIterWaveM, BM//(nbIterWaveM * TM), TM, N//BN, nbIterWaveN, BN//(nbIterWaveN * TN), TN, N//BK, BK)
  a = a.reshape((N//BM, nbIterWaveM, BM//(nbIterWaveM * TM), TM, 1, 1, 1, 1, N//BK, BK)).expand(full_shape)
  b = b.reshape((1, 1, 1, 1, N//BN, nbIterWaveN, BN//(nbIterWaveN * TN), TN, N//BK, BK)).expand(full_shape)
  c = c.reshape((N//BM, nbIterWaveM, BM//(nbIterWaveM * TM), TM, N//BN, nbIterWaveN, BN//(nbIterWaveN * TN), TN, 1, 1))
  As = As.reshape((1, nbIterWaveM, BM//(nbIterWaveM * TM), TM, 1, 1, 1, 1, 1, BK)).expand(full_shape)
  Bs = Bs.reshape((1, 1, 1, 1, 1, nbIterWaveN, BN//(nbIterWaveN * TN), TN, 1, BK)).expand(full_shape)
  A_col = A_col.reshape((1, nbIterWaveM, 1, TM, 1, 1, 1, 1, 1, 1)).expand(full_shape)
  B_row = B_row.reshape((1, 1, 1, 1, 1, nbIterWaveN, 1, TN, 1, 1)).expand(full_shape)

  #                     U1   L2 L3 L4 L5   U6 U7      U9   L10 L11 L12 L13   U14 U15      U17  U18  U19
  expanded_shape = (32, 2,   2, 2, 2, 2,   2, 2,  32, 2,   2,  2,  2,  2,    2,  2,  512, 2,   2,   2)
  assert len(expanded_shape) == 20
  permute_a = list(range(len(expanded_shape)))
  permute_b = permute_a[:]

  # this makes all the global loads match
  # this can also be more simply done by rebinding the RANGEs
  # but sadly, rebinding the RANGEs doesn't work to change the order of the local axes
  permute_a[17:20] = [11,12,13]
  permute_a[11:14] = [17,18,19]
  permute_a[7], permute_a[10] = permute_a[10], permute_a[7]
  permute_a[2:7] = [3,4,5,6,2]

  permute_b[2:16] = [19,9,10,11,17,18,8,2,12,13,14,15,3,4]
  permute_b[17:20] = [5,6,7]

  a_permute   = a.reshape(expanded_shape).permute(tuple(permute_a)).reshape(full_shape)
  As_permute = As.reshape(expanded_shape).permute(tuple(permute_a)).reshape(full_shape)

  b_permute   = b.reshape(expanded_shape).permute(tuple(permute_b)).reshape(full_shape)
  Bs_permute = Bs.reshape(expanded_shape).permute(tuple(permute_b)).reshape(full_shape)

  #out = (a.load() * b.load()).r(Ops.ADD, (8, 9))
  out = (As.load(As_permute.store(a_permute.load())) * Bs.load(Bs_permute.store(b_permute.load()))).r(Ops.ADD, (8, 9))
  #out = (A_col.load(A_col.store(As.load(As.store(a.load())))) * B_row.load(B_row.store(Bs.load(Bs.store(b.load()))))).r(Ops.ADD, (8, 9))

  axis_types = (
    AxisType.GLOBAL, AxisType.UPCAST, AxisType.LOCAL, AxisType.UPCAST,
    AxisType.GLOBAL, AxisType.UPCAST, AxisType.LOCAL, AxisType.UPCAST,
    AxisType.REDUCE, AxisType.REDUCE)

  sink = c.store(out).sink(arg=KernelInfo(name="tg_"+to_colored(full_shape, axis_types), axis_types=axis_types))
  sink = graph_rewrite(sink, merge_views)
  return sink

def hand_spec_kernel3(kernel4=getenv("K4", 0), kernel5=getenv("K5", 0)):
  BLOCK_SIZE = 128 if kernel5 else 256

  nbWaves = BLOCK_SIZE // 32
  WN = 128 if kernel5 else 64
  WM = BN * BM // nbWaves // WN

  nbWaveX = BN // WN
  nbWaveY = BM // WM

  threadIdx_x = UOp(Ops.SPECIAL, dtypes.int, arg=("lidx0", BLOCK_SIZE))
  waveIndex = threadIdx_x // 32
  waveIdx = waveIndex % nbWaveX
  waveIdy = waveIndex // nbWaveX
  indexInWave = threadIdx_x % 32

  nbThreadXPerWave = 8
  nbThreadYPerWave = 4

  idxInWave = indexInWave % nbThreadXPerWave
  idyInWave = indexInWave // nbThreadXPerWave

  nbIterWaveN = WN // (nbThreadXPerWave * TN)
  nbIterWaveM = WM // (nbThreadYPerWave * TM)

  SUBWN = WN // nbIterWaveN
  SUBWM = WM // nbIterWaveM

  # Thread mapping to read BKxBN block from A
  rAIdx = threadIdx_x % BK
  rAIdy = threadIdx_x // BK
  # Thread mapping to read BNxBK block from B
  rBIdx = threadIdx_x % BN
  rBIdy = threadIdx_x // BN

  strideReadB = BLOCK_SIZE // BN
  strideReadA = BLOCK_SIZE // BK
  nbReadsB = BN * BK // BLOCK_SIZE
  nbReadsA = BM * BK // BLOCK_SIZE

  blockIdx_x = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", N//BN))
  blockIdx_y = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx1", N//BM))

  a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1)
  b = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2)
  c = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0)

  A_col = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveM * TM, AddrSpace.REG), arg=0)
  B_row = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveN * TN, AddrSpace.REG), arg=1)

  BM_As_stride = (BM+4) if kernel5 else BM
  As = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BM_As_stride, AddrSpace.LOCAL), arg=0)
  Bs = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BN, AddrSpace.LOCAL), arg=1)

  c_regs = UOp(Ops.DEFINE_REG, dtypes.float.ptr(TM * nbIterWaveM * TN * nbIterWaveN), arg=2)

  i = UOp.range(dtypes.int, c_regs.dtype.size, 16)
  init_store = c_regs[i].store(UOp.const(dtypes.float, 0.0), i)

  if kernel4:
    regA = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbReadsA, AddrSpace.REG), arg=3)
    regB = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbReadsB, AddrSpace.REG), arg=4)

    # initial load from globals into locals (0)
    kId = 0

    # load from globals into locals
    i = UOp.range(dtypes.int, nbReadsB, 0)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId
    Bs_store = Bs[(index_y % BK) * BN + index_x % BN].store(b[N * index_y + index_x].load(), i)

    i = UOp.range(dtypes.int, nbReadsA, 1)
    index_x = rAIdx + kId
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    As_store = As[(index_x % BK) * BM_As_stride + index_y % BM].store(a[N * index_y + index_x].load(), i)

    # iterate over the middle chunk
    kId_range = UOp.range(dtypes.int, N//BK-1, 2)
    kId = kId_range*BK

    barrier = UOp.barrier(As_store, Bs_store)

    # load from globals into registers (next round)
    i = UOp.range(dtypes.int, nbReadsB, 3)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId + BK
    regB_store = regB[i].store(b[N * index_y + index_x].load(), i)

    i = UOp.range(dtypes.int, nbReadsA, 4)
    index_x = rAIdx + kId + BK
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    regA_store = regA[i].store(a[N * index_y + index_x].load(), i)

    def inner_loop(first_range, inp_dep=()):
      # inner unroll
      k = UOp.range(dtypes.int, BK, first_range+0)

      # load from locals into registers
      iterWave = UOp.range(dtypes.int, nbIterWaveN, first_range+1)
      i = UOp.range(dtypes.int, TN, first_range+2)
      index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
      B_row_store = B_row[iterWave*TN + i].store(Bs[k*BN + index].load(*inp_dep), iterWave, i)

      iterWave = UOp.range(dtypes.int, nbIterWaveM, first_range+3)
      i = UOp.range(dtypes.int, TM, first_range+4)
      index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
      A_col_store = A_col[iterWave*TM + i].store(As[k*BM_As_stride + index].load(*inp_dep), iterWave, i)

      # do the GEMM math
      iterWaveM = UOp.range(dtypes.int, nbIterWaveM, first_range+5)
      yt = UOp.range(dtypes.int, TM, first_range+6)
      iterWaveN = UOp.range(dtypes.int, nbIterWaveN, first_range+7)
      xt = UOp.range(dtypes.int, TN, first_range+8)
      x = iterWaveN * TN + xt
      y = iterWaveM * TM + yt
      c_regs_idx = c_regs[y * TN * nbIterWaveN + x]
      # sketchy, this should end the kId_range but it doesn't
      sink = c_regs_idx.store(c_regs_idx.load(init_store) + A_col[y].load(A_col_store) * B_row[x].load(B_row_store),
                              iterWaveM, iterWaveN, yt, xt, k)
      return sink

    # TODO: kId_range should endrange after a barrier
    sink = inner_loop(5, (barrier, regB_store, regA_store)).barrier()

    # load from registers into locals
    i = UOp.range(dtypes.int, nbReadsB, 14)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId + BK
    Bs_store = Bs[(index_y % BK) * BN + index_x % BN].store(regB[i].load(sink), i, kId_range)

    i = UOp.range(dtypes.int, nbReadsA, 15)
    index_x = rAIdx + kId + BK
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    As_store = As[(index_x % BK) * BM_As_stride + index_y % BM].store(regA[i].load(sink), i, kId_range)

    # final iteration without the copy
    sink = inner_loop(16, (UOp.barrier(Bs_store, As_store),))
  else:
    kId_range = UOp.range(dtypes.int, N//BK, 0)
    kId = kId_range*BK

    # load from globals into locals
    i = UOp.range(dtypes.int, nbReadsB, 1)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId
    Bs_store = Bs[(index_y % BK) * BN + index_x % BN].store(b[N * index_y + index_x].load(), i)

    i = UOp.range(dtypes.int, nbReadsA, 2)
    index_x = rAIdx + kId
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    As_store = As[(index_x % BK) * BM_As_stride + index_y % BM].store(a[N * index_y + index_x].load(), i)

    barrier = UOp.barrier(As_store, Bs_store)

    k = UOp.range(dtypes.int, BK, 3)

    # load from locals into registers
    iterWave = UOp.range(dtypes.int, nbIterWaveN, 4)
    i = UOp.range(dtypes.int, TN, 5)
    index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
    B_row_store = B_row[iterWave*TN + i].store(Bs[k*BN + index].load(barrier), iterWave, i)

    iterWave = UOp.range(dtypes.int, nbIterWaveM, 6)
    i = UOp.range(dtypes.int, TM, 7)
    index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
    A_col_store = A_col[iterWave*TM + i].store(As[k*BM_As_stride + index].load(barrier), iterWave, i)

    # do the GEMM math
    iterWaveM = UOp.range(dtypes.int, nbIterWaveM, 8)
    yt = UOp.range(dtypes.int, TM, 9)
    iterWaveN = UOp.range(dtypes.int, nbIterWaveN, 10)
    xt = UOp.range(dtypes.int, TN, 12)
    x = iterWaveN * TN + xt
    y = iterWaveM * TM + yt
    c_regs_idx = c_regs[y * TN * nbIterWaveN + x]
    sink = c_regs_idx.store(c_regs_idx.load(init_store) + A_col[y].load(A_col_store) * B_row[x].load(B_row_store),
                            iterWaveM, iterWaveN, yt, xt, k, kId_range)

  # store c_regs into c
  iterWaveM = UOp.range(dtypes.int, nbIterWaveM, 1000)
  yt = UOp.range(dtypes.int, TM, 1001)
  iterWaveN = UOp.range(dtypes.int, nbIterWaveN, 1002)
  xt = UOp.range(dtypes.int, TN, 1003)
  xOut = blockIdx_x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave
  yOut = blockIdx_y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave
  indexC = N * (yOut + yt) + xOut + xt
  sink = c[indexC].store(c_regs[TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)].load(sink),
                         iterWaveM, iterWaveN, yt, xt)

  return sink.sink(arg=KernelInfo(name="tinygemm"))

if __name__ == "__main__":
  HL = getenv("HL")
  if HL == 2: hprg = top_spec_kernel3()
  elif HL == 1: hprg = hl_spec_kernel3()
  else: hprg = hand_spec_kernel3()
  prg = get_program(hprg, Device.default.renderer)
  print(prg.src)
  if getenv("SRC"): exit(0)
  hrunner = CompiledRunner(prg)

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  hc = Tensor.zeros(N, N).contiguous().realize()

  GlobalCounters.reset()
  with Context(DEBUG=2):
    for _ in range(run_count): tc = (a@b).realize()

  GlobalCounters.reset()
  buffers = [hc.uop.buffer, a.uop.buffer, b.uop.buffer]
  ei = ExecItem(hrunner, buffers)
  with Context(DEBUG=2):
    for _ in range(run_count): ei.run(wait=True)
  err = (hc-tc).square().mean().item()
  print(f"hrunner {err}")
  if err > 1e-06: raise RuntimeError("matmul is wrong!")
