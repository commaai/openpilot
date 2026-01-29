#!/usr/bin/env python
"""
JIT Footguns: Documenting unexpected behavior changes when using @TinyJit

Each test shows behavior that works without JIT but changes with JIT.
Comments marked "should be X!" indicate the intuitively expected value.

SILENT MISMATCHES (highest priority - wrong results, no error):
  class_method_shared_across_instances EASY could check if first arg is self and warn
  slice_assign_requires_realize      MED    assign graph not connected to read during JIT replay
  output_buffer_reuse                MED    performance tradeoff, could add option or better docs
  symbolic_pad_view_frozen           MED    pad view BIND values baked in at capture time
  python_constants_frozen            HARD   inherent to tracing JITs
  conditional_branches_frozen        HARD   inherent to tracing JITs

ERRORS RAISED (lower priority - at least users know):
  item_bakes_in_values               EASY   raises JitError if .item()/.data() accessed during capture
  unrealized_const_input_error       EASY   raises JitError for unrealized const inputs
  non_tensor_outputs_error           EASY   raises JitError if return contains non-Tensor values
  positional_kwargs_cannot_mix       EASY   normalize positional args to kwargs using function signature
  duplicate_inputs_fail              MED    would need to handle aliasing in input_replace
  nested_jit_fails_on_second_call    MED    could fail on first call instead of second
"""
import unittest
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.engine.jit import JitError
from tinygrad.helpers import JIT

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

  def test_graph_input_output_aliasing(self):
    """Test that JIT handles input=output aliasing correctly, simulating LLM generate pattern.

    The LLM generate pattern:
    1. First "session": multiple iterations where output becomes next input
    2. Second "session": starts with a NEW input tensor (not the previous output)

    The bug: GraphRunner computes input_replace during _first_run. If at that time input buffer == output buffer
    (aliasing), it incorrectly includes the output position in input_replace. Later, when a DIFFERENT input
    is passed, the output position gets overwritten with the input, corrupting the computation.

    This requires multiple kernels to trigger because single-kernel JITs don't get graphed ("only one kernel doesn't graph").
    """
    from tinygrad import Device
    if Device[Device.DEFAULT].graph is None or JIT != 1:
      self.skipTest("test requires JIT graph support")

    # Multiple operations to create multiple kernels that get batched into a GraphRunner
    @TinyJit
    def step(x):
      y = (x + 1).realize()  # kernel 1
      z = (y * 2).realize()  # kernel 2
      return z

    # Phase 1: warmup and capture
    a = Tensor([10]).contiguous().realize()
    step(a)  # warmup (cnt=0)
    b = Tensor([20]).contiguous().realize()
    x = step(b)  # capture (cnt=1), x = (20+1)*2 = 42

    # Phase 2: first "session" - iterations where output becomes input (triggers _first_run with aliasing)
    for _ in range(3):
      x = step(x)  # (42+1)*2=86, (86+1)*2=174, (174+1)*2=350
    self.assertEqual(x.item(), 350)

    # Phase 3: second "session" - NEW input tensor (simulates new generate() call)
    # The bug: GraphRunner's input_replace incorrectly includes the output position
    # When new input y is passed, it overwrites the output buffer, using old value (350) instead of new (100)
    y = Tensor([100]).contiguous().realize()
    for _ in range(3):
      y = step(y)  # should be (100+1)*2=202, (202+1)*2=406, (406+1)*2=814
    self.assertEqual(y.item(), 814)  # fails with 1406 if bug exists (uses 350 instead of 100)

  def test_multiple_outputs_same_intermediate(self):
    """Multiple outputs derived from the same intermediate - JIT copies aliased inputs to prevent hazard."""
    @TinyJit
    def f(buf, frame):
      new_buf = buf[1:].cat(frame, dim=0)
      return new_buf.contiguous(), new_buf[:1].contiguous()

    buf = Tensor([[0], [1], [2]]).contiguous().realize()
    for i in range(4):
      frame = Tensor([[10+i]]).contiguous().realize()
      expected_first = buf[1:2].numpy().item()
      new_buf, first = f(buf, frame)
      self.assertEqual(first.numpy().item(), expected_first)
      buf = new_buf

  def test_slice_assign_requires_realize(self):
    """Slice assign then read from same buffer - assign isn't connected to read without explicit realize()."""
    from tinygrad import Variable
    v_pos = Variable("pos", 0, 3)

    # without .realize() after assign, the read doesn't see the assigned values
    cache = Tensor.zeros(4, 4).contiguous().realize()
    @TinyJit
    def f_broken(pos):
      cache[pos:pos+1, :].assign(Tensor.ones(1, 4))
      return cache.sum().realize()
    for i in range(4):
      cache.assign(Tensor.zeros(4, 4)).realize()
      self.assertEqual(f_broken(v_pos.bind(i)).item(), 0.0)  # should be 4.0!

    # workaround: add .realize() after assign
    cache2 = Tensor.zeros(4, 4).contiguous().realize()
    @TinyJit
    def f_fixed(pos):
      cache2[pos:pos+1, :].assign(Tensor.ones(1, 4)).realize()
      return cache2.sum().realize()
    for i in range(4):
      cache2.assign(Tensor.zeros(4, 4)).realize()
      self.assertEqual(f_fixed(v_pos.bind(i)).item(), 4.0)

  def test_symbolic_pad_view_frozen(self):
    """Symbolic pad view has BIND values baked in at capture time. TODO: pad should be captured in jit."""
    from tinygrad import Variable
    a = Tensor.rand(3, 10).realize()

    # broken: pad is a view, BIND values frozen at capture (i=2)
    @TinyJit
    def f_broken(a): return (a+1).pad((None, (0, 10-a.shape[1]))).realize()
    for i in range(1, 5): f_broken(a[:, :Variable("i", 1, 10).bind(i)])
    self.assertEqual(int((f_broken(a[:, :Variable("i", 1, 10).bind(4)])[0] != 0).sum().item()), 2)  # should be 4!

    # workaround: contiguous fuses pad into kernel
    @TinyJit
    def f_fixed(a): return (a+1).pad((None, (0, 10-a.shape[1]))).contiguous().realize()
    for i in range(1, 5): f_fixed(a[:, :Variable("i", 1, 10).bind(i)])
    self.assertEqual(int((f_fixed(a[:, :Variable("i", 1, 10).bind(4)])[0] != 0).sum().item()), 4)

  def test_non_tensor_outputs_error(self):
    @TinyJit
    def f(x, mult): return (x * 2).realize(), mult * 10
    with self.assertRaises(JitError):
      for i in range(3): f(Tensor([i]), i)

  def test_duplicate_inputs_fail(self):
    """JIT cannot handle the same tensor passed as multiple arguments."""
    @TinyJit
    def f(a, b): return (a + b).realize()

    x = Tensor([1, 2, 3])
    with self.assertRaises(JitError):
      f(x, x)

  def test_tensors_in_containers(self):
    @TinyJit
    def f(a, arr): return (a + arr[0]).realize()
    for i in range(4):
      a, b = Tensor([1, 1, 1]).realize(), Tensor([i, i, i]).realize()
      np.testing.assert_array_equal(f(a, [b]).numpy(), [1+i, 1+i, 1+i])

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
    with self.assertRaises(JitError):
      for i in range(1, 5):
        f(base[:, i:i+2])  # different offset each time

  def test_shape_change_after_capture_fails(self):
    """Shapes are locked at capture time."""
    @TinyJit
    def f(a, b): return (a + b).realize()

    f(Tensor.randn(10, 10), Tensor.randn(10, 10))  # warmup
    f(Tensor.randn(10, 10), Tensor.randn(10, 10))  # capture

    with self.assertRaises(JitError):
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

  def test_unrealized_const_input_error(self):
    """Const tensors have no buffer to replace, so JIT raises an error. Even explicit .realize() doesn't help."""
    @TinyJit
    def f(a, b): return (a * b).realize()

    # unrealized const fails
    with self.assertRaises(JitError):
      f(Tensor([1, 2, 3]).realize(), Tensor(2))

    # explicit .realize() on const still fails - const cannot be realized to have a buffer
    @TinyJit
    def g(a, b): return (a * b).realize()
    with self.assertRaises(JitError):
      g(Tensor([1, 2, 3]).realize(), Tensor(2).realize())

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

    with self.assertRaises(JitError):
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

    with self.assertRaises(JitError):
      for _ in range(3):
        f(Tensor([1]), Tensor([2]))

  def test_item_creates_unrealized_return(self):
    """.item() in shape computation raises error during JIT capture."""
    @TinyJit
    def f(x): return Tensor.zeros(x.sum().item())

    f(Tensor([1, 1, 1]))  # warmup
    with self.assertRaises(JitError):
      f(Tensor([1, 1, 1]))  # capture - .item() raises

  def test_item_bakes_in_values(self):
    """.item() during JIT capture raises error (would bake in value)."""
    @TinyJit
    def f(x, mask): return x.masked_select(mask)

    f(Tensor([1, 2, 3, 4]), Tensor([True, False, True, False]))  # warmup
    with self.assertRaises(JitError):
      f(Tensor([1, 2, 3, 4]), Tensor([True, False, True, False]))  # capture - .item() raises

  def test_tolist_bakes_in_values(self):
    """.tolist() raises error during JIT capture (would bake in values)."""
    @TinyJit
    def f(x): return Tensor(x.tolist())

    f(Tensor([1, 2, 3]))  # warmup
    with self.assertRaises(JitError):
      f(Tensor([1, 2, 3]))  # capture - .tolist() raises


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
