import unittest
from tinygrad import Tensor, GlobalCounters, dtypes
from tinygrad.ops import Ops
from tinygrad.helpers import Timing, CI, Profiling, WINO, DEBUG, getenv
from tinygrad.codegen.kernel import Kernel
from tinygrad.codegen.heuristic import hand_coded_optimizations

class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = WINO.value
    WINO.value = 1
  def tearDown(self):
    WINO.value = self.old

  def test_speed(self):
    x = Tensor.empty(1,4,9,9)
    w = Tensor.empty(4,4,3,3)

    with Timing("running conv: "):
      out = Tensor.conv2d(x, w)

    with Timing("scheduling: "):
      sched = out.schedule()

    for i,s in enumerate(sched):
      if s.ast.op is not Ops.SINK: continue
      ops = s.ast.toposort()
      with Timing(f"linearize {i} with {len(ops):4d} ops: "):
        l = Kernel(s.ast)
        l.apply_opts(hand_coded_optimizations(l))
        l.linearize()
      assert len(l.sts) <= 256  # just the current value to prevent regression
      if DEBUG >= 2: print(f"{len(l.sts):4d} shapetrackers with max {max(len(x.views) for x in l.sts)} views")
      for st in l.sts:
        assert len(st.views) <= 2, "too many views in winograd"
        if DEBUG >= 3:
          print(f"{len(st.views):3d} views")
          for v in st.views: print(v)

  def test_profile(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    with Profiling(enabled=not CI, sort='time'):
      out = Tensor.conv2d(x,w).realize()
    out.numpy()

  def test_four_kernels(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    GlobalCounters.reset()
    out = Tensor.conv2d(x,w).realize()
    assert GlobalCounters.kernel_count == 4
    out.numpy()

  @unittest.skipIf(getenv("PTX"), "winograd uses too much in PTX")
  def test_counters(self):
    IC, OC, X, Y = 4,4,9,9
    #OC, IC, X, Y = 512, 256, 8, 8
    x,w = Tensor.rand(1,IC,Y,X).realize(), Tensor.rand(OC,IC,3,3).realize()
    GlobalCounters.reset()
    Tensor.conv2d(x,w).realize()
    ops_wino, mem_wino = GlobalCounters.global_ops, GlobalCounters.global_mem
    WINO.value = 0
    GlobalCounters.reset()
    Tensor.conv2d(x,w).realize()
    ops_normal, mem_normal = GlobalCounters.global_ops, GlobalCounters.global_mem

    ops_ratio, mem_ratio = ops_wino/ops_normal, mem_wino/mem_normal
    print(f"ops: normal {ops_normal:9d} wino {ops_wino:9d} ratio {ops_ratio:.2f}")
    print(f"mem: normal {mem_normal:9d} wino {mem_wino:9d} ratio {mem_ratio:.2f}")
    self.assertLess(ops_ratio, 2.6)  # TODO: there's issues with factorization now
    self.assertLess(mem_ratio, 10)

  def test_dtype(self):
    IC, OC, X, Y = 4,4,9,9
    x,w = Tensor.empty(1,IC,Y,X), Tensor.empty(OC,IC,3,3)
    self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.default_float)

    x,w = Tensor.empty(1,IC,Y,X,dtype=dtypes.half), Tensor.empty(OC,IC,3,3,dtype=dtypes.half)
    self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.half)

if __name__ == '__main__':
  unittest.main(verbosity=2)
