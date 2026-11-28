import unittest
from tinygrad import Tensor, Variable, Context, Device
from tinygrad.helpers import cpu_events
from tinygrad.codegen import schedule_cache

def schedule_one():
  Tensor([1]).schedule_linear()

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_var_vals(self):
    v = Variable('pos', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    t = x + Tensor(v.bind(42))
    _, externs = t.linear_with_externs()
    self.assertEqual({x.expr: x.arg.val for x in externs if x.is_bound_var}, {'pos': 42})

  def test_view_offset_not_in_cache_key(self):
    # every use of a contiguous view is the same input, so slices at different offsets share one schedule
    x = Tensor.arange(16).float().clone().realize()
    schedule_cache.clear()
    for start in (1, 2, 3):
      v = x[start:start+4]
      (v.contiguous() + v).realize()
      self.assertEqual(sum(isinstance(k, bytes) for k in schedule_cache), 1)

  def test_disable_schedule_cache(self):
    schedule_cache.clear()

    # test write
    with Context(SCACHE=0): schedule_one()
    self.assertEqual(len(schedule_cache), 0)
    with Context(SCACHE=1):
      schedule_one()
      schedule_one()
    self.assertEqual(len(schedule_cache), 2)  # one body schedule and one final LINEAR

    # test read
    with Context(PROFILE=1):
      cpu_events.clear()
      with Context(SCACHE=0): schedule_one()
      num_events_no_cache = len(cpu_events)

      cpu_events.clear()
      with Context(SCACHE=1): schedule_one()
      num_events_cache = len(cpu_events)
    self.assertLess(num_events_cache, num_events_no_cache)

class TestMethodCache(unittest.TestCase):
  def setUp(self):
    self.backup_compiler = Device[Device.DEFAULT].compiler.compile_cached
  def tearDown(self):
    Device[Device.DEFAULT].compiler.compile_cached = self.backup_compiler

  def test_nested_methodcache(self):
    a,b,c,d = Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])
    ((a+b)+(a+b)).realize()
    Device[Device.DEFAULT].compiler.compile_cached = None
    ((c+d)+(c+d)).realize()

  def test_nested_methodcache_swap(self):
    a,b,c,d = Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])
    ((a+b)+(c+d)).realize()
    Device[Device.DEFAULT].compiler.compile_cached = None
    ((c+d)+(a+b)).realize()

if __name__ == "__main__":
  unittest.main()
