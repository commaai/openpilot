# schedule tests that pass on NULL backend (no copyout needed)
import unittest
from tinygrad import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.helpers import DEBUG, Context
from tinygrad.engine.realize import CompiledRunner, run_schedule

class KernelCountException(Exception): pass
def check_schedule(t:Tensor|list[Tensor]|UOp, allowed:int, to_prerealize:list[Tensor]|None=None, filter_sink=True):
  if to_prerealize:
    with Context(DEBUG=0, TRACK_MATCH_STATS=0): Tensor.realize(*to_prerealize)
  if isinstance(t, Tensor): sched = t.schedule()
  elif isinstance(t, list) and isinstance(t[0], Tensor): sched = Tensor.schedule(*t)
  else:
    assert isinstance(t, UOp), f"can't schedule {t}"
    sched = Tensor(t).schedule()
  # test lowering all the ExecItems
  for si in sched: si.lower()
  kernel_cnt = len([si for si in sched if isinstance(si.prg, CompiledRunner) or not filter_sink])
  if kernel_cnt != allowed:
    print(f"SCHEDULE ISSUE, expecting {allowed} got {kernel_cnt}")
    if DEBUG >= 3:
      for i,s in enumerate(sched):
        print("kernel", i+1)
        print(s.ast)
    raise KernelCountException(f"{kernel_cnt} != {allowed}")
  return sched

class TestBufferUOp(unittest.TestCase):
  # BUFFER has a ShapeTracker of shape=(n,) and stride=(1,)
  def test_buffer_has_buffer(self):
    buf = Tensor.empty(10)
    self.assertIsNotNone(buf.uop.buffer)
    self.assertEqual(buf.uop.shape, (10,))
    # the device Buffer remains unallocated until it's we run the schedule
    self.assertFalse(buf.uop.buffer.is_allocated())
    add = buf+1
    sched = add.schedule()
    self.assertFalse(buf.uop.buffer.is_allocated())
    run_schedule(sched)
    self.assertTrue(buf.uop.buffer.is_allocated())

  def test_buffer_has_unique_buffer(self):
    buf = Tensor.empty(10)
    buf1 = buf.uop.buffer
    buf2 = buf.uop.buffer
    self.assertIs(buf1, buf2)

  # we also allow VIEW(BUFFER) to access the underlying device Buffer, as long as it's contiguous
  def test_buffer_view_allowed(self):
    add = Tensor.empty(1, 1)+Tensor.empty(1, 1)
    add.realize()
    self.assertIsNotNone(add.uop.buffer)
    self.assertEqual(add.uop.shape, (1, 1))

  def test_buffer_view_not_allowed(self):
    permuted_view = Tensor.empty(1, 2, 3).permute(0, 2, 1)
    with self.assertRaisesRegex(AssertionError, "can only be RESHAPE"):
      permuted_view.uop.buffer # cannot access Buffer of a non contiguous VIEW

  def test_buffer_only_after_realize(self):
    a = Tensor([1])+Tensor([2])
    # accessing realized will return None
    self.assertIsNone(a.uop.realized)
    # accessing Buffer will assert
    with self.assertRaisesRegex(AssertionError, "must be BUFFER"):
      a.uop.buffer # there is no BUFFER on an unrealized ADD
    # Buffer only exists once we realize it
    a.realize()
    self.assertIsNotNone(a.uop.buffer)

  def test_const_does_not_realize(self):
    a = Tensor(1)+Tensor(2)
    run_schedule(check_schedule(a, 0))
    self.assertIsNone(a.uop.base.realized)

  def test_var_does_not_realize(self):
    a = Tensor(UOp.variable("a", 0, 10).bind(1))
    run_schedule(check_schedule(a, 0))
    self.assertIsNone(a.uop.base.realized)

  def test_unused_var_not_in_var_vals(self):
    # unused variable should not appear in var_vals even when there's other work
    a = Tensor(UOp.variable("unused", 0, 10).bind(1))
    b = Tensor.empty(3) + 1
    _, var_vals = Tensor.schedule_with_vars(a, b)
    self.assertEqual(var_vals, {})
    self.assertIsNone(a.uop.base.realized)

  def test_view_does_not_realize(self):
    a = Tensor.randn(1, 4).expand(4, 4)
    a.realize()
    self.assertEqual(a.uop.base.realized.size, 4)
    a2 = a.contiguous().realize()
    self.assertEqual(a2.uop.base.realized.size, 16)

class TestContiguous(unittest.TestCase):
  def test_contiguous_buffer(self):
    a = Tensor.empty(4)
    b = a.contiguous()
    check_schedule(b, 0)

  def test_contiguous_buffer_view(self):
    a = Tensor.empty(4)
    b = a.reshape((2, 2)).contiguous()
    check_schedule(b, 0)

  def test_non_contiguous_buffer_view(self):
    a = Tensor.empty(4, 1)
    b = a.expand((4, 4)).contiguous()
    check_schedule(b, 1)

  def test_size_change_buffer_view(self):
    a = Tensor.empty(4)
    b = a.reshape((1, 1, 4)).shrink(((0, 1), (0, 1), (0, 3))).contiguous()
    check_schedule(b, 1)

  def test_double_contiguous_realizes_once(self):
    a = Tensor.empty(4, 1)
    b = a.expand((4, 4)).contiguous().contiguous()
    check_schedule(b, 1)

  def test_view_does_not_realize(self):
    a = Tensor.empty(4)
    b = a.expand((4, 4))
    check_schedule(b, 0)
    self.assertEqual(b.uop.base.buffer.size, 4)

  def test_contiguous_view_realizes(self):
    a = Tensor.empty(4)
    b = a.expand((4, 4)).contiguous()
    check_schedule(b, 1)
    self.assertEqual(b.uop.base.buffer.size, 16)

class TestSimpleSchedule(unittest.TestCase):
  def test_reduce_doesnt_split(self):
    a = Tensor.empty(16,16).sum(axis=1)
    a1 = a.reshape(4,4)
    a2 = a.reshape(16,1,1)
    self.assertEqual(len(Tensor.schedule(a1, a2)), 1)

if __name__ == '__main__':
  unittest.main(verbosity=2)
