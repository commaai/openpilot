import unittest
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.uop.ops import AxisType

# TODO: write a clean version of this
from test.backend.test_linearizer import helper_linearizer_opt

class TestKernelOpts(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_local_and_grouped_reduce(self):
    N = 128
    Tensor.manual_seed(1882)
    a = Tensor.rand(4, 4, N, N)
    b = Tensor.rand(4, 4, N)
    r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    helper_linearizer_opt(r, [
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL))],
      [Opt(OptOps.SPLIT, 0, (8, AxisType.LOCAL))],
      [Opt(OptOps.SPLIT, 0, (16, AxisType.LOCAL))], # Checking how it works with locals
      [Opt(OptOps.SPLIT, 1, (2, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 1, (32, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 1, (64, AxisType.GROUP_REDUCE, True))], # Checking how it works with grouped reduce
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 0, (16, AxisType.LOCAL)), Opt(OptOps.SPLIT, 2, (16, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 0, (32, AxisType.LOCAL)), Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE, True))],
      # Checking how it works with locals + grouped reduce
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 2, (64, AxisType.GROUP_REDUCE, True))],
      # Checking how it works with locals + grouped reduce + upcasts
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE, True)), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),
       Opt(OptOps.SPLIT, 4, (4, AxisType.UNROLL))],
      # many local + many group
      [Opt(OptOps.SPLIT, 1, (2, AxisType.GROUP_REDUCE)), Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE)),
       Opt(OptOps.SPLIT, 3, (2, AxisType.GROUP_REDUCE)), Opt(OptOps.SPLIT, 4, (2, AxisType.GROUP_REDUCE))],
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL))] * 4,
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE)),
       Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (2, AxisType.GROUP_REDUCE)),
       Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 6, (2, AxisType.GROUP_REDUCE)),
       Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 8, (2, AxisType.GROUP_REDUCE))],
    ])

  def test_upcasts(self):
    N = 16
    Tensor.manual_seed(1772)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST))], # Checking how it works with upcasts
    ])

  def test_full_upcast(self):
    Tensor.manual_seed(1772)
    a = Tensor.rand(4)
    b = Tensor.rand(4)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))], # Checking how it works with upcasts
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_matmul(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    helper_linearizer_opt(r, [
      [Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST))], # Checking how it works with upcasts
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL))],
      [Opt(OptOps.SPLIT, 1, (32, AxisType.LOCAL))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (4, AxisType.LOCAL))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (32, AxisType.LOCAL))],
      [Opt(OptOps.SPLIT, 0, (16, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (8, AxisType.LOCAL))], # Checking how it works with locals
      [Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 2, (32, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 2, (32, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 2, (4, AxisType.UNROLL))], # Checking how it works with grouped_reduce
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (32, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 0, (8, AxisType.LOCAL)), Opt(OptOps.SPLIT, 3, (32, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 0, (8, AxisType.LOCAL)),
       Opt(OptOps.SPLIT, 4, (4, AxisType.GROUP_REDUCE, True))], # Checking how it works with local+grouped_reduce
      # Checking all together
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (8, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 4, (4, AxisType.UNROLL)), Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)),
       Opt(OptOps.SPLIT, 1, (2, AxisType.UPCAST))],
      # Full global upcast + local
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (8, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 4, (4, AxisType.UNROLL)), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST))],
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_double_reduce(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(8, N, 8, N)
    r = a.sum(axis=(1,3))
    helper_linearizer_opt(r, [
      # openCL / DEV=CL is 256 max threads
      [Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE, True))], [Opt(OptOps.SPLIT, 2, (32, AxisType.GROUP_REDUCE, True))],
      # Checking how it works with 1 grouped_reduce.
      [Opt(OptOps.SPLIT, 3, (2, AxisType.GROUP_REDUCE, True))], [Opt(OptOps.SPLIT, 3, (32, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE, True)), Opt(OptOps.SPLIT, 4, (2, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 2, (16, AxisType.GROUP_REDUCE, True)), Opt(OptOps.SPLIT, 4, (2, AxisType.GROUP_REDUCE, True))],
      [Opt(OptOps.SPLIT, 2, (4, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 4, (64, AxisType.GROUP_REDUCE, True))], # Checking how it works with 2 grouped_reduces.
      [Opt(OptOps.SPLIT, 2, (16, AxisType.GROUP_REDUCE, True)), Opt(OptOps.SPLIT, 4, (2, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 2, (4, AxisType.UNROLL))],
      # Checking how it works with 2 grouped_reduces + upcasts.
      [Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE, True)), Opt(OptOps.SPLIT, 4, (32, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 4, (4, AxisType.UNROLL))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (4, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 6, (4, AxisType.GROUP_REDUCE, True))],
      # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (2, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 6, (32, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 5, (4, AxisType.UNROLL))],
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (8, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 6, (4, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (8, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 6, (4, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)), Opt(OptOps.SPLIT, 4, (4, AxisType.UNROLL)),
       Opt(OptOps.SPLIT, 5, (4, AxisType.UNROLL))], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 4, (4, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 6, (4, AxisType.GROUP_REDUCE, True)),
       Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)), Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))], # No globals
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  def test_tensor_core_opts(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST))], # check upcasts
      [Opt(OptOps.SPLIT, 4, (2, AxisType.UNROLL))], # check unroll
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 5, (2, AxisType.UNROLL))], # check combo of unroll and upcast
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 6, (2, AxisType.UNROLL))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 6, (4, AxisType.UNROLL))],
      [Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))], # check permutations
      [Opt(OptOps.SPLIT, 4, (2, AxisType.UNROLL)), Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 5, (2, AxisType.UNROLL)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 4, (2, AxisType.UNROLL)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)),
       Opt(OptOps.SPLIT, 6, (4, AxisType.UNROLL))],
    ], apply_tc=True, atol=atol, rtol=rtol)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_tensor_core_opts_locals(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [Opt(OptOps.SPLIT, 4, (0, AxisType.UNROLL))], # check full unroll of reduce with locals
      [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL))], # check local
      [Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 6, (4, AxisType.UNROLL)),
       Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL))],
      [Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 6, (2, AxisType.UNROLL)),
       Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))],
    ], apply_tc=True, atol=atol, rtol=rtol)

  def test_padto_matmul(self):
    N = 17
    Tensor.manual_seed(289)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 2, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.PADTO, 2, 32)],
      # can optimize further post PADTO
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (2, AxisType.UPCAST)),],
    ])

  def test_padto_upcasted_not_ok(self):
    N = 4
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 1, (0, AxisType.UPCAST))],
      [Opt(OptOps.SPLIT, 2, (0, AxisType.UNROLL))],
      [Opt(OptOps.PADTO, 0, 8)],
      [Opt(OptOps.PADTO, 1, 8)],
      [Opt(OptOps.PADTO, 2, 8)],
    ])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST)), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.SPLIT, 1, (0, AxisType.UPCAST)), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.SPLIT, 2, (0, AxisType.UNROLL)), Opt(OptOps.PADTO, 2, 8)]])

  def test_padto_sum_ok(self):
    N = 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).realize().shrink(((0, 17), (0, 17))) * 100
    b = (Tensor.rand(N, N) < 0.5).realize().shrink(((0, 17), (0, 17)))

    helper_linearizer_opt(a.sum(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),],
    ])
    helper_linearizer_opt(a.sum(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),],
    ])

    for axis in (0, 1):
      helper_linearizer_opt(a.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(a.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
      # TODO: why?
      if Device.DEFAULT != "WEBGPU":
        helper_linearizer_opt(b.sum(0, dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
        helper_linearizer_opt(b.sum(1, dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])

    # having unsafe ops after sum is fine
    helper_linearizer_opt(a.sum().exp(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.sum(0).exp(), [[Opt(OptOps.PADTO, 1, 32)],])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_padto_group_full_unroll_sum(self):
    a = Tensor.ones(2, 28, 4096).realize()
    out = ((a * 0.5).float().square()).sum(axis=(0, 2))
    opts_to_apply = [Opt(OptOps.SPLIT, 2, (256, AxisType.GROUP_REDUCE, True)), Opt(OptOps.PADTO, 3, 32), Opt(OptOps.SPLIT, 3, (0, AxisType.UNROLL)),
                     Opt(OptOps.SPLIT, 0, (7, AxisType.UPCAST))]
    helper_linearizer_opt(out, [opts_to_apply], check_default_opt=False)

  def test_padto_unrolled_sum(self):
    a = Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17).clone().realize()
    for amt in (4, 0):
      helper_linearizer_opt(a.sum(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (amt, AxisType.UNROLL))]])

  def test_padto_unrolled_max(self):
    a = (Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17) - 100).clone().realize()
    for amt in (4, 0):
      helper_linearizer_opt(a.max(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (amt, AxisType.UNROLL))]])

  def test_padto_unrolled_upcast(self):
    a = Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17).clone().realize()
    helper_linearizer_opt(a.sum(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (0, AxisType.UNROLL)),
                                      Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))]])

  @unittest.skipUnless(any(tc.dtype_in in (dtypes.half, dtypes.float) for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                       "test requires half or float tensor cores")
  def test_tc_shape_padded(self):
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in in (dtypes.half, dtypes.float))
    Tensor.manual_seed(3)
    a, b = Tensor.rand(17, 23, dtype=tc.dtype_in).realize(), Tensor.rand(23, 29, dtype=tc.dtype_in).realize()
    with Context(ALLOW_TF32=1):
      helper_linearizer_opt(a.matmul(b, dtype=tc.dtype_out), [[Opt(OptOps.TC, 0, (-1, 2, 2))]], check_default_opt=False, atol=3e-2, rtol=1e-3)

  @unittest.skipUnless(any(tc.dtype_in in (dtypes.half, dtypes.float) for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                       "test requires half or float tensor cores")
  def test_tc_padto_full_upcast(self):
    # a fully upcast pad lane makes a WMMA operand entirely Invalid
    tc = next(tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in in (dtypes.half, dtypes.float))
    Tensor.manual_seed(3)
    a, b = Tensor.rand(17, 23, dtype=tc.dtype_in).realize(), Tensor.rand(23, 29, dtype=tc.dtype_in).realize()
    with Context(ALLOW_TF32=1):
      helper_linearizer_opt(a.matmul(b, dtype=tc.dtype_out),
                            [[Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.PADTO, 0, 4), Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST))]],
                            check_default_opt=False, atol=3e-2, rtol=1e-3)

  def test_padto_nested_reduce(self):
    a = (Tensor.arange(2*3, dtype=dtypes.float).reshape(2, 3) + 1).clone().realize()  # [[1, 2, 3], [4, 5, 6]]
    # the pad gate has the outer reduce's range, the inner reduce must not resolve it with its own identity
    pad_outer = [[Opt(OptOps.PADTO, 1, 4)]]
    helper_linearizer_opt(a.max(1).sum(0), pad_outer, wanna_output=[[3+6]])
    helper_linearizer_opt((-a).sum(1).max(0), pad_outer, wanna_output=[[-6]])
    helper_linearizer_opt(a.prod(1).sum(0), pad_outer, wanna_output=[[6+120]])
    # both reduce axes padded: the outer clause lifts out, the inner clause is the inner reduce's identity
    helper_linearizer_opt(a.max(1).sum(0), [[Opt(OptOps.PADTO, 0, 4), Opt(OptOps.PADTO, 1, 4)]], wanna_output=[[3+6]])

  def test_padto_unrolled_prod(self):
    a = (Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17) / 100 + 1).clone().realize()
    helper_linearizer_opt(a.prod(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (0, AxisType.UNROLL)),
                                       Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))]])

  def test_padto_arg(self):
    a = Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17).clone().realize()
    for arg in (-4, 0, 1, True):
      with self.assertRaises(KernelOptError):
        helper_linearizer_opt(a.sum(1), [[Opt(OptOps.PADTO, 1, arg)]])

  def test_padto_sum(self):
    N = 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).shrink(((0, 17), (0, 17))).exp()
    helper_linearizer_opt(a.exp().sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.exp().sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

    b = a < 1
    helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_max(self):
    N = 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one axis
    a = -Tensor.rand(N, N).shrink(((0, 17), (0, 17))) * 100

    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),],
    ])
    helper_linearizer_opt(a.max(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),],
    ])

    helper_linearizer_opt(a.max(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.max(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_where(self):
    Tensor.manual_seed(0)
    N = 17
    a = (Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1).where(1, 0).int()
    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),],
    ])

  def test_padto_where_multioutput(self):
    Tensor.manual_seed(0)
    N = 17
    r = Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1
    a0 = r.where(1, 0).int()
    a1 = r.where(2, 0).int()
    helper_linearizer_opt([a0.max(0), a1.max(0)], [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.SPLIT, 0, (8, AxisType.UPCAST)),],
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_color_shapes_with_local(self):
    N = 32
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    opts_shapes = [
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL))], [("blue",16),("blue",32),("cyan",2),("red",32)]),
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (2, AxisType.GROUP_REDUCE))],
       [("blue",16),("blue",32),("cyan",2),("green",2),("red",16)]),
      # check to ensure local_dims are stable for full UNROLL of the first reduce
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (0, AxisType.UNROLL))], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.SPLIT, 2, (0, AxisType.UNROLL)),Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL))], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      # check behavior for full UNROLL on an existing GROUP
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (0, AxisType.GROUP_REDUCE)),Opt(OptOps.SPLIT, 3, (2, AxisType.UNROLL))],
       [("blue",16),("blue",32),("cyan",2),("green",16),("magenta",2)]),
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (0, AxisType.GROUP_REDUCE)),Opt(OptOps.SPLIT, 3, (0, AxisType.UNROLL))],
       [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.SPLIT, 2, (0, AxisType.GROUP_REDUCE)),Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 2, (0, AxisType.UNROLL))],
       [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.SPLIT, 2, (2, AxisType.GROUP_REDUCE)),Opt(OptOps.SPLIT, 2, (0, AxisType.UNROLL))],
       [("blue",32),("blue",32),("red",16),("magenta",2)]),
    ]
    helper_linearizer_opt(r, [x[0] for x in opts_shapes], color_sizes=[x[1] for x in opts_shapes])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_arange_opts(self):
    a = Tensor.arange(128).clone()
    # NOTE: arange no longer has reduce ops available for opt
    helper_linearizer_opt(a, [
      [Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))],
      [Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST))],
    ])

  def test_double_sum_group(self):
    a = Tensor.rand(4, 4, 4)
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.SPLIT, 0, (16, AxisType.GROUP_REDUCE, True))],])
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.SPLIT, 1, (4, AxisType.UNROLL)), Opt(OptOps.SPLIT, 0, (16, AxisType.GROUP_REDUCE, True))],])
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.SPLIT, 1, (4, AxisType.GROUP_REDUCE, True)), Opt(OptOps.SPLIT, 1, (16, AxisType.GROUP_REDUCE, True))],])

if __name__ == '__main__':
  unittest.main()
