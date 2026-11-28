import unittest
from tinygrad import Tensor, UOp, Device, GlobalCounters
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import KernelInfo
from test.helpers import assert_kernel_count
from test.runtime.test_custom_kernel import custom_add_one_kernel

# a gated index pushed through movement ops keeps gating the access
class TestGatedIndexMovement(unittest.TestCase):
  def test_gated_tensor_index_valid(self):
    y, k = Tensor([7]).realize().expand(3).uop, Tensor([2, 0]).realize().uop
    self.assertEqual(Tensor(y.index(k.valid(k.eq(0)))).tolist(), [0, 7])

  def test_gated_store_2d(self):
    # the valid on one index of the 2d C gates the whole store, only j == 0 writes C[i, 0]
    def kernel(C:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(4, 1)
      return C[i.valid(j.eq(0)), 0].store((j+1).cast(C.dtype)).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4, 4), fxn=kernel)[0][:, 0].tolist(), [1.]*4)

  @unittest.skipIf(not Device[Device.DEFAULT].renderer.has_shared, "LOCAL buffer needs shared memory")
  def test_gated_local_store_2d(self):
    # the valid on one index of the 2d LOCAL tmp gates the whole store, only j == 0 writes tmp[i, 0]
    def kernel(C:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(4, 1)
      tmp = UOp.placeholder((4, 4), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
      st = tmp[i.valid(j.eq(0)), 0].store((j+1).cast(dtypes.float)).end(j)
      return C[i].store(tmp.after(st)[i, 0]).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4), fxn=kernel)[0].tolist(), [1.]*4)

  def test_gated_store_partial_reshape(self):
    def kernel(C:UOp) -> UOp:
      i, j = UOp.range(2, 0), UOp.range(3, 1)
      dst = C.reshape(2, 2, 4).index(i.valid(j.eq(0)), 0)
      return dst.store((j+10).cast(C.dtype).reshape(1).expand(4)).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.zeros(4, 4).contiguous(), fxn=kernel)[0].tolist(), [[10.]*4, [0.]*4]*2)

  def test_gated_store_reshape_indexed_clause(self):
    def kernel(C:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(2, 1)
      return C.reshape(4, 2, 2).index(i.valid(j.eq(0)), j, 0).store(7.).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.zeros(16).contiguous(), fxn=kernel)[0].tolist(), [7., 0., 0., 0.]*4)

  def test_gated_load_2d(self):
    # the valid on one index of the 2d A gates the whole load, it is 0 where j != 0
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(4, 1)
      return C[i, j].store(A[i.valid(j.eq(0)), 0]).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor.arange(16).reshape(4, 4).float().contiguous().realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4, 4), a, fxn=kernel)[0].tolist(), a[:, :1].pad(((0, 0), (0, 3))).tolist())

  def test_gated_load_expand(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(5, 0), UOp.range(4, 1)
      return C[i, j].store(A.reshape(1, 4).expand(4, 4).pad(((0, 1), (0, 0)))[i, j]).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor([1., 2., 3., 4.]).realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(5, 4), a, fxn=kernel)[0].tolist(), [[1., 2., 3., 4.]]*4+[[0.]*4])

  def test_gated_load_expand_double_pad(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(6, 0), UOp.range(4, 1)
      v = A.reshape(1, 4).expand(4, 4).pad(((1, 0), (0, 0))).pad(((1, 0), (0, 0)))
      return C[i, j].store(v[i, j]).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor([1., 2., 3., 4.]).realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(6, 4), a, fxn=kernel)[0].tolist(), [[0.]*4]*2+[[1., 2., 3., 4.]]*4)

  def test_gated_load_expand_bool(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      i = UOp.range(3, 0)
      return C[i].store(A.reshape(()).expand(3)[i.valid(i.eq(0))].where(7., 2.)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(3), Tensor([True]).realize(), fxn=kernel)[0].tolist(), [7., 2., 2.])

  def test_gated_load_reshape(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(3, 0), UOp.range(4, 1)
      return C[i, j].store(A.reshape(4, 4).index(i.valid(j.eq(0))+1, 0)).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor([float(i) for i in range(16)]).realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(3, 4), a, fxn=kernel)[0].tolist(), [[v, 0., 0., 0.] for v in (4., 8., 12.)])

  def test_gated_load_reshape_indexed_clause(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(4, 0), UOp.range(2, 1)
      return C[i, j].store(A.reshape(4, 2, 2).index(i.valid(j.eq(0)), j, 0)).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor([float(i) for i in range(16)]).realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(4, 2), a, fxn=kernel)[0].tolist(), [[v, 0.] for v in (0., 4., 8., 12.)])

  def test_gated_load_reshape_nested(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      i, j = UOp.range(3, 0), UOp.range(4, 1)
      return C[i, j].store(A.reshape(4, 4).index((i.valid(j.eq(0))+1).valid(j < 2), 0)).end(i, j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor([float(i) for i in range(16)]).realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(3, 4), a, fxn=kernel)[0].tolist(), [[v, 0., 0., 0.] for v in (4., 8., 12.)])

  def test_gated_load_reshape_stack_nested(self):
    def kernel(C:UOp, A:UOp) -> UOp:
      j = UOp.range(3, 0)
      ix = UOp.stack(UOp.const(0).valid(j.eq(0)).valid(j < 2), UOp.const(1))
      return C[j].store(A.reshape(4, 4).index(ix, 0)).end(j).sink(arg=KernelInfo(opts_to_apply=()))
    a = Tensor([float(i+1) for i in range(16)]).realize()
    self.assertEqual(Tensor.custom_kernel(Tensor.empty(3, 2), a, fxn=kernel)[0].tolist(), [[1., 5.], [0., 5.], [0., 5.]])

# an invalid view has no contiguous offset
@unittest.skipIf(Device.DEFAULT in {"WEBGPU", "CL"}, "subbuffer not supported")
class TestContiguousViewOffsetInvalid(unittest.TestCase):
  def test_expand_shrink_invalid(self): self.assertIsNone(UOp.empty(1,4).expand(2,4).pad(((0,1),(0,0)))[2].contiguous_view_offset())
  def test_scalar_expand_shrink_invalid(self): self.assertIsNone(UOp.empty(1).expand(2).pad(((0,1),))[2:3].contiguous_view_offset())

class TestCustomKernelInput(unittest.TestCase):
  def _test_mop(self, mop_fxn, max_kernels):
    # default: input is BUFFER
    x = mop_fxn(Tensor.arange(32).clone().realize())
    y = Tensor.custom_kernel(Tensor.empty_like(x), x, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    y.realize()
    assert_kernel_count(max_kernels)
    self.assertEqual(y.tolist(), x.add(1).tolist())
    # same test with @function, input is PARAM
    from tinygrad import function
    x0 = Tensor.arange(32).clone().realize()
    @function(precompile=True)
    def run(a:Tensor) -> Tensor:
      xv = mop_fxn(a)
      y = Tensor.invalids(*xv.shape, dtype=xv.dtype, device=a.device)
      return Tensor.custom_kernel(y, xv, fxn=custom_add_one_kernel)[0]
    GlobalCounters.reset()
    y = run(x0).realize()
    assert_kernel_count(max_kernels)
    self.assertEqual(y.tolist(), mop_fxn(x0).add(1).tolist())

  def test_reshape(self): self._test_mop(lambda x: x.reshape(16, 2), max_kernels=1)
  def test_permute(self): self._test_mop(lambda x: x.reshape(4, 8).T, max_kernels=2)
  def test_double_permute(self): self._test_mop(lambda x: x.reshape(4, 8).T.T, max_kernels=1)
  def test_shrink(self): self._test_mop(lambda x: x[:4], max_kernels=1)
  def test_pad(self): self._test_mop(lambda x: x[:4].pad(((0, 4),)), max_kernels=2)
  def test_flip(self): self._test_mop(lambda x: x.flip(0), max_kernels=2)
  def test_offset_shrink(self): self._test_mop(lambda x: x[4:8], max_kernels=2)
  def test_2d_shrink(self): self._test_mop(lambda x: x.reshape(4, 8)[:, 2:6], max_kernels=2)
  def test_expand(self): self._test_mop(lambda x: x.reshape(16, 2)[:, :1].expand(16, 2), max_kernels=2)

if __name__ == '__main__':
  unittest.main()
