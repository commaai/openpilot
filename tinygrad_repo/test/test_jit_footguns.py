#!/usr/bin/env python
"""
JIT Footguns: Documenting unexpected behavior changes when using @TinyJit

Each test shows behavior that works without JIT but changes with JIT.
Comments marked "should be X!" indicate the intuitively expected value.

SILENT MISMATCHES (highest priority - wrong results, no error):
  tensors_in_containers_ignored      EASY   only checks t.__class__ is Tensor, could scan lists/dicts
  non_tensor_outputs_frozen          EASY   could warn/error if return contains non-Tensor values
  class_method_shared_across_instances EASY could check if first arg is self and warn
  output_buffer_reuse                MED    performance tradeoff, could add option or better docs
  python_constants_frozen            HARD   inherent to tracing JITs
  conditional_branches_frozen        HARD   inherent to tracing JITs

ERRORS RAISED (lower priority - at least users know):
  positional_kwargs_cannot_mix       EASY   normalize positional args to kwargs using function signature
  duplicate_inputs_fail              MED    would need to handle aliasing in input_replace
  nested_jit_fails_on_second_call    MED    could fail on first call instead of second
"""
import unittest
import numpy as np
from tinygrad import Tensor, TinyJit

class TestJitFootguns(unittest.TestCase):

  def test_output_buffer_reuse(self):
    """Output tensors share buffer after capture - old references get overwritten."""
    @TinyJit
    def f(x): return x.sum().realize()

    r1 = f(Tensor([1, 1]))  # warmup
    r2 = f(Tensor([2, 2]))  # capture
    r3 = f(Tensor([3, 3]))  # jit exec

    self.assertEqual(r1.item(), 2)  # warmup result independent
    self.assertEqual(r3.item(), 6)  # latest is correct
    self.assertEqual(r2.item(), 6)  # should be 4! (overwritten by r3)

  def test_output_buffer_workaround(self):
    """Use .clone().realize() to get independent copies."""
    @TinyJit
    def f(x): return x.sum().realize()

    r1 = f(Tensor([1, 1])).clone().realize()
    r2 = f(Tensor([2, 2])).clone().realize()
    r3 = f(Tensor([3, 3])).clone().realize()

    self.assertEqual([r1.item(), r2.item(), r3.item()], [2, 4, 6])

  def test_non_tensor_outputs_frozen(self):
    """Non-tensor return values are frozen at capture time."""
    @TinyJit
    def f(x, mult): return (x * 2).realize(), mult * 10

    # collect results, copying tensor values immediately (buffer reuse!)
    results = []
    for i in range(5):
      t, s = f(Tensor([i]), i)
      results.append((t.item(), s))

    # tensor outputs work correctly
    self.assertEqual([r[0] for r in results[2:]], [4, 6, 8])
    # scalar outputs frozen at capture (i=1) - should be 20, 30, 40!
    self.assertEqual([r[1] for r in results[2:]], [10, 10, 10])

  def test_duplicate_inputs_fail(self):
    """JIT cannot handle the same tensor passed as multiple arguments."""
    @TinyJit
    def f(a, b): return (a + b).realize()

    x = Tensor([1, 2, 3])
    with self.assertRaises(AssertionError):
      f(x, x)

  def test_tensors_in_containers_ignored(self):
    """Tensors inside lists/dicts are not tracked as inputs."""
    @TinyJit
    def f(a, arr): return (a + arr[0]).realize()

    results = []
    for i in range(4):
      a, b = Tensor([1, 1, 1]).realize(), Tensor([i, i, i]).realize()
      results.append(f(a, [b]).numpy().copy())

    np.testing.assert_array_equal(results[0], [1, 1, 1])  # warmup
    np.testing.assert_array_equal(results[1], [2, 2, 2])  # capture
    np.testing.assert_array_equal(results[2], [2, 2, 2])  # should be [3,3,3]!
    np.testing.assert_array_equal(results[3], [2, 2, 2])  # should be [4,4,4]!

  def test_nested_jit_fails_on_second_call(self):
    """Nested JIT works on first call but fails on second."""
    @TinyJit
    def inner(t): return t + 1
    @TinyJit
    def outer(t): return inner(t) * 3

    self.assertEqual(outer(Tensor([1])).realize().item(), 6)  # works!
    with self.assertRaises(RuntimeError):
      outer(Tensor([2])).realize()  # fails

  def test_implicit_inputs_need_realize(self):
    """Closure tensors must be realized before JIT call."""
    x = Tensor([0])

    @TinyJit
    def f(): return (x * 2).realize()

    for i in range(5):
      x.assign(Tensor([i])).realize()  # must realize!
      self.assertEqual(f().item(), i * 2)

  def test_views_with_different_offsets_fail(self):
    """JIT requires consistent tensor views across calls."""
    @TinyJit
    def f(a): return (a + 1).realize()

    base = Tensor.randn(10, 10).realize()
    with self.assertRaises(AssertionError):
      for i in range(1, 5):
        f(base[:, i:i+2])  # different offset each time

  def test_shape_change_after_capture_fails(self):
    """Shapes are locked at capture time."""
    @TinyJit
    def f(a, b): return (a + b).realize()

    f(Tensor.randn(10, 10), Tensor.randn(10, 10))  # warmup
    f(Tensor.randn(10, 10), Tensor.randn(10, 10))  # capture

    with self.assertRaises(AssertionError):
      f(Tensor.randn(20, 20), Tensor.randn(20, 20))

  def test_python_constants_frozen(self):
    """Python variables inside JIT use capture-time values."""
    mult = 1

    @TinyJit
    def f(x): return (x * mult).realize()

    results = []
    for i in range(5):
      mult = i + 1
      results.append(f(Tensor([10])).item())

    self.assertEqual(results[0], 10)   # warmup, mult=1
    self.assertEqual(results[1], 20)   # capture, mult=2
    self.assertEqual(results[2], 20)   # should be 30!
    self.assertEqual(results[3], 20)   # should be 40!

  def test_conditional_branches_frozen(self):
    """Only the branch taken during capture runs thereafter."""
    @TinyJit
    def f(x, use_square):
      if use_square:
        return (x * x).realize()
      return (x * 2).realize()

    f(Tensor([3]), True)   # warmup
    f(Tensor([3]), False)  # capture (False branch)

    result = f(Tensor([3]), True)  # passing True but False branch runs
    self.assertEqual(result.item(), 6)  # should be 9!

  def test_positional_kwargs_cannot_mix(self):
    """Must use same calling convention after capture."""
    @TinyJit
    def f(a, b): return (a + b).realize()

    f(Tensor([1]), Tensor([2]))  # warmup with positional
    f(Tensor([1]), Tensor([2]))  # capture with positional

    with self.assertRaises(AssertionError):
      f(a=Tensor([3]), b=Tensor([4]))  # kwargs fail

  def test_class_method_shared_across_instances(self):
    """JIT on instance methods is shared at class level."""
    class Model:
      def __init__(self, scale):
        self.scale = Tensor([scale])
      @TinyJit
      def forward(self, x):
        return (x * self.scale).realize()

    m1, m2 = Model(2), Model(3)

    m1.forward(Tensor([5]))  # warmup
    m1.forward(Tensor([5]))  # capture with m1.scale=2

    self.assertEqual(m1.forward(Tensor([5])).item(), 10)
    self.assertEqual(m2.forward(Tensor([5])).item(), 10)  # should be 15!

  def test_side_effects_only_during_capture(self):
    """Function body not executed during JIT replay."""
    call_count = [0]

    @TinyJit
    def f(x):
      call_count[0] += 1
      return (x * 2).realize()

    f(Tensor([1]))  # warmup
    f(Tensor([2]))  # capture
    self.assertEqual(call_count[0], 2)

    f(Tensor([3]))
    f(Tensor([4]))
    f(Tensor([5]))
    self.assertEqual(call_count[0], 2)  # still 2, not 5!

  def test_nothing_realized_fails(self):
    """Must JIT at least one kernel."""
    @TinyJit
    def f(a, b): return None

    with self.assertRaises(AssertionError):
      for _ in range(3):
        f(Tensor([1]), Tensor([2]))


class TestJitCorrectBehavior(unittest.TestCase):
  """Behaviors that work correctly - documented for clarity."""

  def test_random_regenerates(self):
    """Random tensors regenerate each call."""
    @TinyJit
    def f(x):
      return (x + Tensor.rand(3)).realize()

    f(Tensor([0, 0, 0]))  # warmup
    f(Tensor([0, 0, 0]))  # capture

    results = {tuple(f(Tensor([0, 0, 0])).numpy().tolist()) for _ in range(5)}
    self.assertEqual(len(results), 5)

  def test_unrealized_return_auto_realized(self):
    """Unrealized return tensors are auto-realized."""
    @TinyJit
    def f(a, b): return a + b  # no explicit realize

    for _ in range(5):
      a, b = Tensor.randn(10), Tensor.randn(10)
      np.testing.assert_allclose(f(a, b).numpy(), a.numpy() + b.numpy(), atol=1e-5)

  def test_kwargs_order_doesnt_matter(self):
    """Kwargs are sorted by name, so order doesn't matter."""
    @TinyJit
    def f(first, second): return (first / second).realize()

    for _ in range(3):
      a, b = Tensor.randn(10), Tensor.randn(10) + 1
      np.testing.assert_allclose(f(second=b, first=a).numpy(), a.numpy() / b.numpy(), atol=1e-4)
      np.testing.assert_allclose(f(first=a, second=b).numpy(), a.numpy() / b.numpy(), atol=1e-4)

  def test_input_mutation_consistent(self):
    """Input mutation via assign works consistently."""
    @TinyJit
    def f(x):
      x += 1
      x.realize()
      return x

    a = Tensor([0]).contiguous().realize()
    for _ in range(5):
      f(a)
    self.assertEqual(a.item(), 5)


if __name__ == '__main__':
  unittest.main()
