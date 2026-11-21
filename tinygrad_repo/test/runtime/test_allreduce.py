import unittest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from test.helpers import KernelCountException
from tinygrad.engine.realize import run_linear

class TestRingAllReduce(unittest.TestCase):
  def test_hierarchy(self):
    ds = tuple(f"CPU:{i}" for i in range(4))
    with Context(ALL2ALL=1, ALLREDUCE_NODE_NDEVS=2): # two nodes of 2
      for size in (1, 17):
        x = (Tensor.arange(4 * size, dtype=dtypes.int32).reshape(4, size) % 13).realize()
        self.assertEqual(x.shard(ds, axis=0).sum(0).tolist(), x.sum(0).tolist())

  def test_schedule_all2all(self):
    with Context(ALL2ALL=2):
      N = 4
      M = N*100
      ds = tuple(f"CPU:{i}" for i in range(N))
      x = Tensor.arange(N*M, dtype=dtypes.float).reshape(N, M)
      t = (x*x).clone().shard(ds, axis=0).realize()
      out = t.sum(0).mul(2.).contiguous()
      linear, var_vals = out.linear_with_vars()
      copies = [si for si in linear.src if si.src[0].op is Ops.STORE]
      sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
      # N*(N-1) copies for input and output
      copy_count = N*(N-1)*2
      if len(copies) != copy_count: raise KernelCountException(copy_count, len(copies))
      # N*(N-1) shrinks from other devices becoming contigs, N ALU, N extra contig, reassembly (cat), and mul
      sink_count = (N*(N-1))+(N)+(N)+(1)+(1)
      if len(sinks) != sink_count: raise KernelCountException(sink_count, len(sinks))
      # correctness
      run_linear(linear, var_vals)
      expected = [2*sum((d*M+i)**2 for d in range(N)) for i in range(M)]
      dev_nums = Tensor.arange(1, N+1, dtype=dtypes.float).reshape(N, 1).expand(N, M).shard(ds, axis=0)
      shards = out.reshape(1, M).expand(N, M)+dev_nums
      self.assertListEqual(shards.tolist(), [[x+d+1 for x in expected] for d in range(N)])

  def test_symbolic_shape(self):
    rows = UOp.variable("rows", 1, 4).bind(3)
    t = Tensor.ones(4, 4).shard(("CPU:0", "CPU:1"), axis=1).realize()
    out = t[:rows].sum(1).realize()
    self.assertEqual(out.shape, (rows,))
    self.assertTrue((out == 4).all().item())

  def test_symbolic_shape_2d(self):
    rows = UOp.variable("rows", 1, 4).bind(3)
    t = Tensor.ones(4, 4, 2).shard(("CPU:0", "CPU:1"), axis=1).realize()
    out = t[:rows].sum(1).realize()
    self.assertEqual(out.shape, (rows, 2))
    self.assertTrue((out == 4).all().item())

  def test_correct_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.ones(N, N*100).contiguous().shard(ds, axis=0).realize()
      out = t.sum(0)
      self.assertListEqual(out.tolist(), [4]*N*100)

if __name__ == '__main__':
  unittest.main()
