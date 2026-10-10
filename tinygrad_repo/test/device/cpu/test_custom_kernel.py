import unittest
from tinygrad import Tensor, UOp, Context, GlobalCounters
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.renderer.cstyle import ClangRenderer
from test.helpers import assert_kernel_count

class TestCustomKernel(unittest.TestCase):
  @Context(DEV="CPU")
  def test_simple_from_source(self):
    a = Tensor.arange(4).clone().realize()
    src = ClangRenderer.kernel_typedef + " test_src(int* restrict a) { a[0] = 1; }"
    def custom_src_kernel(A:UOp, B:UOp) -> UOp:
      sink = UOp.sink(A, arg=KernelInfo(name="test_src"))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=src),))
    a = Tensor.custom_kernel(a.reshape(2, 2).clone(), a.reshape(2, 2).T, fxn=custom_src_kernel)[0]
    self.assertEqual(a.tolist(), [[1, 1], [2, 3]])

  @Context(DEV="CPU")
  def test_simple_from_source_alt(self):
    a = Tensor.arange(4).clone().realize()
    src = ClangRenderer.kernel_typedef + " copy(int* restrict out, int* restrict in) { for (int i = 0; i < 4; i++) out[i] = in[i]; }"
    def custom_src_kernel(out:UOp, inp:UOp) -> UOp:
      sink = UOp.sink(out, inp, arg=KernelInfo(name="copy"))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=src),))
    out = Tensor.custom_kernel(Tensor.empty_like(a), a+1, fxn=custom_src_kernel)[0]
    GlobalCounters.reset()
    out.realize()
    assert_kernel_count(2)
    self.assertEqual(out.tolist(), [1, 2, 3, 4])

if __name__ == "__main__": unittest.main()
