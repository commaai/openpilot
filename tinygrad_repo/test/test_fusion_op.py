import unittest
import time
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import lower_schedule_item, run_schedule

class TestFusionOp(unittest.TestCase):
  def test_contiguous_add(self):
    def test(contig=False):
      bt = Tensor(np.arange(16), dtype=dtypes.float32).reshape(4,4)
      x = bt.permute(1,0)
      if contig: x = x.contiguous()
      return (x.permute(1,0) + bt).data()
    assert test() == test(True)

  def test_expand_fuse(self):
    bt = Tensor(np.ones((10, 1)), dtype=dtypes.float32)
    out = (bt*2).expand(10,10).sum(1)
    sched = out.schedule()
    run_schedule(sched)
    outd = out.tolist()
    assert all(x == 20.0 for x in outd)

  def test_recursive_add(self):
    st = time.perf_counter()
    a = Tensor([1,2,3,4])
    for _ in range(24): a = a + a
    sched = a.schedule()
    ei = lower_schedule_item(sched[-1])
    self.assertLess(time.perf_counter()-st, 2.0)
    assert len(ei.prg.p.src.splitlines()) < 250

  def test_recursive_add_cmp(self):
    st = time.perf_counter()
    a = Tensor([1,2,3,4])
    for _ in range(24): a = a + a
    sched1 = a.schedule()
    b = Tensor([1,2,3,4])
    for _ in range(24): b = b + b
    sched2 = b.schedule()
    c = Tensor([1,2,3,4])
    for _ in range(23): c = c + c
    sched3 = c.schedule()
    self.assertEqual(sched1[-1].ast, sched2[-1].ast)
    with self.assertRaises(AssertionError): self.assertEqual(sched1[-1].ast, sched3[-1].ast)
    self.assertLess(time.perf_counter()-st, 2.0)

  def test_recursive_pad(self):
    st = time.perf_counter()
    val = 1.0
    a = Tensor(val)
    for _ in range(24): a = Tensor.stack(a, a)[0]
    sched = a.schedule()
    self.assertEqual(len(sched), 1)
    self.assertLess(time.perf_counter()-st, 2.0)

  def test_recursive_reshape(self):
    st = time.perf_counter()
    a = Tensor.empty(32, 32).realize()
    b = Tensor.empty(16, 2).realize()
    r = a.sum(1)
    for _ in range(24): r = r.reshape(16, 2) + b
    sched = r.schedule()
    self.assertEqual(len(sched), 1)
    self.assertLess(time.perf_counter()-st, 2.0)

if __name__ == '__main__':
  unittest.main(verbosity=2)
