#!/usr/bin/env python
import gc, weakref, contextlib
import unittest
import numpy as np
from tinygrad.tensor import Tensor, all_tensors
from test.helpers import run_schedule

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

@contextlib.contextmanager
def gc_frees_tensors():
  # every collection frees one Tensor, as a collected cycle does when it held the last reference
  held = [Tensor.empty(1) for _ in range(100)]
  def free_one(phase, info):
    if held and phase == "start": held.pop()
  threshold = gc.get_threshold()
  gc.callbacks.append(free_one)
  gc.set_threshold(1)
  try: yield
  finally:
    gc.set_threshold(*threshold)
    gc.callbacks.remove(free_one)

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

  def test_schedule_gc_with_inputs(self):
    x = Tensor.ones(256).contiguous().realize()
    with assert_freed(x.uop.buffer):
      y = x+Tensor.ones(256).contiguous()
      del x
      run_schedule(*y.linear_with_externs())
    with assert_freed(y.uop.buffer):
      del y

  def test_toposort_blocks_gc(self):
    x = Tensor.ones(4,4).contiguous().realize()+1
    with assert_freed(x.uop.src[0].buffer):
      x.uop.toposort()
      del x

  def test_buffer_ownership(self):
    a = Tensor.empty(10).realize()
    real_buf = a.uop.buffer
    with assert_freed(real_buf):
      self.assertFalse(real_buf.is_allocated())
      a.realize()
      self.assertIs(a.uop.buffer, real_buf)
      del a.uop # the Buffer object is still held by real_buf
      del real_buf

  def test_cyclic_gc_releases_buffer(self):
    a = Tensor.empty(10).realize()
    with assert_freed(a, a.uop.buffer):
      holder = [a]
      holder.append(holder)
      del a, holder
    self.assertTrue(all(r() is not None for r in all_tensors))

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

  def test_setitem_while_gc_frees_tensors(self):
    a = Tensor.zeros(4).contiguous().realize()
    with gc_frees_tensors(): a[0:1] = 1

  def test_backward_while_gc_frees_tensors(self):
    a = Tensor.ones(4)
    with gc_frees_tensors(): (a*a).sum().backward()

if __name__ == '__main__':
  unittest.main()
