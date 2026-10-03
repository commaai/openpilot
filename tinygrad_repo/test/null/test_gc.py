#!/usr/bin/env python
import gc, weakref, contextlib
import unittest
import numpy as np
from tinygrad.engine.realize import run_linear
from tinygrad.tensor import Tensor

def _allocations_of_type(t):
  ret = 0
  for x in gc.get_objects():
    try:
      if isinstance(x, t): ret += 1
    except ReferenceError:
      pass
  return ret

def tensors_allocated():
  gc.collect()
  return _allocations_of_type(Tensor)

@contextlib.contextmanager
def assert_freed(*objects):
  refs = [weakref.ref(obj) for obj in objects]
  del objects
  yield
  gc.collect()
  for ref in refs: assert ref() is None, f"{ref()} was not freed"

class TestGC(unittest.TestCase):
  def test_gc(self):
    Tensor.manual_seed(0)
    base = tensors_allocated()
    a = Tensor.rand(4, 4)
    b = Tensor.zeros(4, 4)
    (a*b).mean().backward()
    assert (tensors_allocated()-base > 0)
    del a,b
    assert (tensors_allocated()-base == 2) # one for Tensor._device_rng_counters, and one for Tensor._device_seeds
    Tensor.manual_seed(0)

  def test_gc_complex(self):
    Tensor.manual_seed(0)
    base = tensors_allocated()
    a = Tensor(np.zeros((4, 4), dtype=np.float32))
    b = Tensor.rand(4, 4)
    assert (tensors_allocated()-base == 4)
    (a*b).mean().backward()
    assert (tensors_allocated()-base == 6)
    del b
    assert (tensors_allocated()-base == 4)
    b = Tensor(np.zeros((4, 4), dtype=np.float32))
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert (tensors_allocated()-base == 6)
    del b
    assert (tensors_allocated()-base == 4)
    Tensor.manual_seed(0)

  def test_schedule_gc(self):
    x = Tensor.ones(256).contiguous().realize()
    y = Tensor.ones(5, 5).contiguous()
    y.schedule_linear()
    with assert_freed(x.uop.buffer, y.uop.buffer):
      del x, y

  def test_schedule_gc_with_inputs(self):
    x = Tensor.ones(256).contiguous().realize()
    with assert_freed(x.uop.buffer):
      y = x+Tensor.ones(256).contiguous()
      del x
      run_linear(*y.linear_with_vars())
    with assert_freed(y.uop.buffer):
      del y

  def test_toposort_blocks_gc(self):
    x = Tensor.ones(4,4).contiguous().realize()+1
    with assert_freed(x.uop.src[0].buffer):
      x.uop.toposort()
      del x

  def test_buffer_ownership(self):
    a = Tensor.empty(10)
    real_buf = a.uop.buffer
    with assert_freed(real_buf):
      self.assertFalse(real_buf.is_allocated())
      a.realize()
      self.assertIs(a.uop.arg.buffer, real_buf)
      del a.uop # the Buffer object is still held by real_buf
      del real_buf

  def test_assign_keeps_buffer(self):
    a = Tensor.full((4,), 1.).contiguous()
    a.realize()
    real_buf = a.uop.buffer
    with assert_freed(real_buf):
      a.assign(Tensor.full((4,), 2.))
      # assign writes in place: the AFTER still references the same Buffer
      self.assertIs(a.uop.src[0].buffer, real_buf)
      a.realize()
      self.assertIs(a.uop.buffer, real_buf)
      del a
      self.assertTrue(real_buf.is_allocated()) # the Buffer object is still held here
      del real_buf

if __name__ == '__main__':
  unittest.main()
