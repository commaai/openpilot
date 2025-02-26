#!/usr/bin/env python
import gc, inspect
import unittest
import numpy as np
from tinygrad.device import Buffer
from tinygrad.engine.realize import run_schedule
from tinygrad.ops import UOp
from tinygrad.tensor import Tensor

def tensors_allocated():
  gc.collect()
  return sum([isinstance(x, Tensor) for x in gc.get_objects()])

def bufs_allocated():
  gc.collect()
  return sum([isinstance(x, Buffer) for x in gc.get_objects()])

class TestGC(unittest.TestCase):

  def test_gc(self):
    Tensor.manual_seed(0)
    base = tensors_allocated()
    a = Tensor.rand(4, 4, requires_grad=True)
    b = Tensor.zeros(4, 4, requires_grad=True)
    (a*b).mean().backward()
    assert (tensors_allocated()-base > 0)
    del a,b
    assert (tensors_allocated()-base == 2) # one for Tensor._device_rng_counters, and one for Tensor._device_seeds
    Tensor.manual_seed(0)

  def test_gc_complex(self):
    Tensor.manual_seed(0)
    base = tensors_allocated()
    a = Tensor(np.zeros((4, 4), dtype=np.float32), requires_grad=True)
    b = Tensor.rand(4, 4, requires_grad=True)
    assert (tensors_allocated()-base == 4)
    (a*b).mean().backward()
    assert (tensors_allocated()-base == 6)
    del b
    assert (tensors_allocated()-base == 4)
    b = Tensor(np.zeros((4, 4), dtype=np.float32), requires_grad=True)
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert (tensors_allocated()-base == 6)
    del b
    assert (tensors_allocated()-base == 4)
    Tensor.manual_seed(0)

  def test_schedule_gc(self):
    init = bufs_allocated()
    x = Tensor.ones(256).contiguous().realize()
    y = Tensor.ones(5, 5).contiguous()
    y.schedule()
    del x
    del y
    self.assertEqual(bufs_allocated()-init, 0)

  def test_schedule_gc_with_inputs(self):
    init = bufs_allocated()
    x = Tensor.ones(256).contiguous().realize()
    y = x+Tensor.ones(256).contiguous()
    ys = y.schedule()
    del x
    run_schedule(ys)
    np.testing.assert_equal(y.numpy(), np.full((256,), 2))
    self.assertEqual(bufs_allocated()-init, 1)
    del y
    self.assertEqual(bufs_allocated()-init, 0)

  def test_toposort_blocks_gc(self):
    init = bufs_allocated()
    x = Tensor.ones(4,4).contiguous().realize()+1
    self.assertEqual(bufs_allocated()-init, 1)
    # try commenting this part out, it's green!
    x.lazydata.toposort
    del x
    if bufs_allocated()-init != 0:
      print(inspect.getclosurevars(UOp.toposort.fget))
      raise AssertionError(f"never gced {[x for x in gc.get_objects() if isinstance(x, Buffer)]}")

if __name__ == '__main__':
  unittest.main()
