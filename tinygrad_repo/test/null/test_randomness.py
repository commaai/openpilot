import unittest
from tinygrad import dtypes, Tensor
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import Ops

class TestRandomness(unittest.TestCase):
  def test_threefry_doesnt_use_long(self):
    # Use a renderer without native THREEFRY or 64-bit pointer arithmetic in the UOps.
    renderer = ClangRenderer(Target("CPU", arch="x86_64,x86-64"))
    linear = Tensor.rand(20).schedule_linear()
    for call in linear.src:
      ast = call.src[0]
      if ast.op is Ops.SINK:
        prg = to_program(ast, renderer=renderer)
        for u in tuple(prg.src[1].src):
          self.assertNotIn(u.dtype, {dtypes.long, dtypes.ulong}, msg=f"long found in {prg.src[0].arg.name}")

  def test_rand_like(self):
    empty = Tensor.empty((80, 44))
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_randn_like(self):
    empty = Tensor.empty((80, 44))
    rand = Tensor.randn_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_zero_shape(self):
    empty = Tensor.empty(0, 20)
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_more_dims(self):
    empty = Tensor.empty((1, 2, 3, 4, 5, 6))
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_dtype(self):
    empty = Tensor.empty((80, 44), dtype=dtypes.float16)
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

    empty = Tensor.empty((80, 44))
    rand = Tensor.rand_like(empty, dtype=dtypes.float16)
    assert rand.shape == empty.shape
    assert rand.dtype == dtypes.float16
    assert rand.device == empty.device

  def test_randn_like_dtype(self):
    empty = Tensor.empty((80, 44), dtype=dtypes.float16)
    rand = Tensor.randn_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

    empty = Tensor.empty((80, 44))
    rand = Tensor.randn_like(empty, dtype=dtypes.float16)
    assert rand.shape == empty.shape
    assert rand.dtype == dtypes.float16
    assert rand.device == empty.device

  def test_randn_device(self):
    self.assertEqual(Tensor.randn(3,3,device="NULL").device, "NULL")

if __name__ == '__main__':
  unittest.main()
