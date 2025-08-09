import unittest
from tinygrad import TinyJit, Tensor

# The JIT functions as a "capturing" JIT.
# Whatever kernels ran in the JIT the second run through the function will be the kernels that will run from then on.
# Explicit inputs to the function are updated in the JIT graph to the new inputs.

# JITs have four tensor types
#  1. Tensors that are explicit in the input, aka what's passed in. TODO: support lists/dicts/classes, anything get_state works on
#  2. Tensors that are explicit in the output, aka what's returned. TODO: same as above
#  3. Tensors that are implicit in the input as a closure.
#  4. Tensors that are implicit in the output because they were assigned to and realized.

# explicit inputs and outputs are realized on their way in and out of the JIT
# there's a whole bunch of edge cases and weirdness here that needs to be tested and clarified.

class TestJitCases(unittest.TestCase):
  def test_explicit(self):
    # this function has an explicit input and an explicit output
    @TinyJit
    def f(x:Tensor):
      ret:Tensor = x*2
      return ret

    for i in range(5):
      out = f(Tensor([i]))
      self.assertEqual(out.item(), i*2)

  def test_implicit_input(self):
    # x is the implicit input (like a weight)
    x = Tensor([0])

    # this function has an implicit input and an explicit output
    @TinyJit
    def f():
      ret:Tensor = x*2
      return ret

    for i in range(5):
      # NOTE: this must be realized here, otherwise the update doesn't happen
      # if we were explicitly tracking the implicit input Tensors, we might not need this realize
      x.assign(Tensor([i])).realize()
      out = f()
      self.assertEqual(out.item(), i*2)

  def test_implicit_output(self):
    # out is the implicit output (it's assigned to)
    out = Tensor([0])

    # this function has an explicit input and an implicit output
    @TinyJit
    def f(x:Tensor):
      # NOTE: this must be realized here
      # if we were explicitly tracking the implicit output Tensors, we might not need this realize
      out.assign(x*2).realize()

    for i in range(5):
      f(Tensor([i]))
      self.assertEqual(out.item(), i*2)

  def test_implicit_io(self):
    # x is the implicit input (like a weight)
    # out is the implicit output (it's assigned to)
    x = Tensor([0])
    out = Tensor([0])

    # this function has an implicit input and an implicit output
    @TinyJit
    def f():
      out.assign(x*2).realize() # NOTE: this must be realized here

    for i in range(5):
      x.assign(Tensor([i])).realize()
      f()
      self.assertEqual(out.item(), i*2)

if __name__ == '__main__':
  unittest.main()
