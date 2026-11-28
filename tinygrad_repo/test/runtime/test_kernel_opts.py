import unittest
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.uop.ops import AxisType
from tinygrad.helpers import DEV

# TODO: write a clean version of this
from test.runtime.test_linearizer import helper_linearizer_opt

class TestKernelOpts(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_grouped_reduce_with_local_upcast_padto(self):
    Tensor.manual_seed(7)
    a = Tensor.rand(7, 11, 13)
    helper_linearizer_opt(a.sum((1, 2)) + a.max((1, 2)), [
      [Opt(OptOps.SPLIT, 0, (0, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (11, AxisType.UPCAST)),
       Opt(OptOps.SPLIT, 2, (0, AxisType.LOCAL, True)), Opt(OptOps.PADTO, 2, 32)],
    ])
    b = Tensor.rand(17, 19)
    helper_linearizer_opt(b.flip(0).pad(((2, 3), (0, 0))).sum(0), [
      [Opt(OptOps.SPLIT, 1, (0, AxisType.LOCAL, True)), Opt(OptOps.PADTO, 0, 8),
       Opt(OptOps.SPLIT, 0, (12, AxisType.UPCAST)), Opt(OptOps.SPLIT, 0, (0, AxisType.LOCAL))],
    ])
    x, w = Tensor.rand(1, 3, 15, 15), Tensor.rand(4, 3, 3, 3)
    helper_linearizer_opt(x.conv2d(w, padding=1, stride=2), [
      [Opt(OptOps.SPLIT, 5, (0, AxisType.LOCAL, True)), Opt(OptOps.SPLIT, 1, (0, AxisType.LOCAL))],
    ])

  @unittest.skipIf(Device.DEFAULT == "AMD" and DEV.interface.startswith("MOCK"), "AMD emulator doesn't support direct pc read and write")
  def test_unrolled_padded_cumsum(self):
    Tensor.manual_seed(7)
    a = Tensor.rand(13, 17)
    helper_linearizer_opt(a.cumsum(1), [
      [Opt(OptOps.SPLIT, 2, (0, AxisType.UPCAST)), Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST)), Opt(OptOps.PADTO, 0, 4)],
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

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_matmul_upcast_group(self):
    Tensor.manual_seed(1552)
    helper_linearizer_opt(Tensor.rand(16, 128)@Tensor.rand(128, 16),
                          [[Opt(OptOps.SPLIT, 2, (32, AxisType.LOCAL, True)), Opt(OptOps.SPLIT, 2, (4, AxisType.UPCAST))]])

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
      [Opt(OptOps.SPLIT, 2, (0, AxisType.UPCAST))],
      [Opt(OptOps.PADTO, 0, 8)],
      [Opt(OptOps.PADTO, 1, 8)],
      [Opt(OptOps.PADTO, 2, 8)],
    ])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.SPLIT, 0, (0, AxisType.UPCAST)), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.SPLIT, 1, (0, AxisType.UPCAST)), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.SPLIT, 2, (0, AxisType.UPCAST)), Opt(OptOps.PADTO, 2, 8)]])

  def test_padto_cmpeq_sum(self):
    a, b = Tensor(list(range(17))), Tensor([i%3 for i in range(17)])
    # The extra lanes compare 0 == 0, but must not contribute to the reduction.
    helper_linearizer_opt((a == b).sum(), [[Opt(OptOps.PADTO, 0, 32)]])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipIf(Device.DEFAULT == "CL", "flaky segfault in rusticl")
  def test_padto_group_full_unroll_sum(self):
    a = Tensor.ones(2, 28, 4096).realize()
    out = ((a * 0.5).float().square()).sum(axis=(0, 2))
    opts_to_apply = [Opt(OptOps.SPLIT, 2, (256, AxisType.LOCAL, True)), Opt(OptOps.PADTO, 3, 32), Opt(OptOps.SPLIT, 3, (0, AxisType.UPCAST)),
                     Opt(OptOps.SPLIT, 0, (7, AxisType.UPCAST))]
    helper_linearizer_opt(out, [opts_to_apply], check_default_opt=False)

  def test_padto_unrolled_sum(self):
    a = Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17).clone().realize()
    for amt in (4, 0):
      helper_linearizer_opt(a.sum(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (amt, AxisType.UPCAST))]])

  def test_padto_unrolled_max(self):
    a = (Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17) - 100).clone().realize()
    for amt in (4, 0):
      helper_linearizer_opt(a.max(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (amt, AxisType.UPCAST))]])

  def test_padto_unrolled_upcast(self):
    a = Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17).clone().realize()
    helper_linearizer_opt(a.sum(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (0, AxisType.UPCAST)),
                                      Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))]])

  def test_padto_nested_reduce(self):
    a = (Tensor.arange(2*3, dtype=dtypes.float).reshape(2, 3) + 1).clone().realize()  # [[1, 2, 3], [4, 5, 6]]
    # the pad gate has the outer reduce's range, the inner reduce must not resolve it with its own identity
    pad_outer = [[Opt(OptOps.PADTO, 1, 4)]]
    helper_linearizer_opt(a.max(1).sum(0), pad_outer, wanna_output=[[3+6]])
    helper_linearizer_opt((-a).sum(1).max(0), pad_outer, wanna_output=[[-6]])
    helper_linearizer_opt(a.prod(1).sum(0), pad_outer, wanna_output=[[6+120]])
    # both reduce axes padded: the outer clause lifts out, the inner clause is the inner reduce's identity
    helper_linearizer_opt(a.max(1).sum(0), [[Opt(OptOps.PADTO, 0, 4), Opt(OptOps.PADTO, 1, 4)]], wanna_output=[[3+6]])

  def test_padto_unindexed_reduce(self):
    # a repeat/cat along the reduced axis leaves a reduce range no buffer index uses
    a = Tensor.arange(7*5, dtype=dtypes.float).reshape(7, 5).clone().realize()
    helper_linearizer_opt(a.repeat((3, 16)).sum(1), [[Opt(OptOps.PADTO, 2, 32)]])
    helper_linearizer_opt(a.repeat((1, 16)).sum(1), [[Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.PADTO, 1, 3)]])
    helper_linearizer_opt(a.cat(a, dim=1).sum(1), [[Opt(OptOps.PADTO, 1, 4)]])
    a = Tensor.full((7, 5), 2.0).clone().realize()
    helper_linearizer_opt(a.repeat((1, 4)).prod(1), [[Opt(OptOps.PADTO, 1, 3)]])

  def test_padto_reduce_identity(self):
    a = Tensor.arange(7*5, dtype=dtypes.float).reshape(7, 5).clone().realize()
    helper_linearizer_opt(a.sum(1), [[Opt(OptOps.PADTO, 1, 8), Opt(OptOps.PADTO, 1, 16)]])
    helper_linearizer_opt((a > 0).all(1), [[Opt(OptOps.PADTO, 1, 8)]])

  def test_padto_masked_reduce(self):
    # a where with a defined false arm gives the padded iterations a value
    a = Tensor.arange(7*17, dtype=dtypes.float).reshape(7, 17).clone().realize()
    m = (Tensor.arange(7).reshape(7, 1) % 2 == 0).expand(7, 17)
    helper_linearizer_opt(m.where(a, 1.0).sum(1), [[Opt(OptOps.PADTO, 1, 8)]])
    helper_linearizer_opt(m.where(a, 1.0).sum(1), [[Opt(OptOps.PADTO, 1, 32)]])
    helper_linearizer_opt((Tensor.arange(17).reshape(1, 17) < 5).expand(7, 17).where(a, 1.0).sum(1), [[Opt(OptOps.PADTO, 1, 8)]])
    helper_linearizer_opt(m[:, :11].where(Tensor.ones(7, 11), 2.0).prod(1), [[Opt(OptOps.PADTO, 1, 8)]])

  def test_padto_unrolled_prod(self):
    a = (Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17) / 100 + 1).clone().realize()
    helper_linearizer_opt(a.prod(1), [[Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SPLIT, 1, (0, AxisType.UPCAST)),
                                       Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST))]])

  def test_padto_arg(self):
    a = Tensor.arange(4*17, dtype=dtypes.float).reshape(4, 17).clone().realize()
    for arg in (-4, 0, 1, True):
      with self.assertRaises(KernelOptError):
        helper_linearizer_opt(a.sum(1), [[Opt(OptOps.PADTO, 1, arg)]])

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
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (2, AxisType.LOCAL))],
       [("blue",16),("blue",32),("cyan",2),("green",2),("red",16)]),
      # check to ensure local_dims are stable for full upcast of the first reduce
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (0, AxisType.UPCAST))], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      # check behavior for full upcast on an existing group
      ([Opt(OptOps.SPLIT, 0, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (0, AxisType.LOCAL)),Opt(OptOps.SPLIT, 3, (2, AxisType.UPCAST))],
       [("blue",16),("blue",32),("cyan",2),("green",16),("magenta",2)]),
      ([Opt(OptOps.SPLIT, 2, (2, AxisType.LOCAL)),Opt(OptOps.SPLIT, 2, (0, AxisType.UPCAST))],
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

  def test_top_split_non_reduce_axis(self):
    # top puts the new range as the outer factor of any split
    helper_linearizer_opt(Tensor.rand(64, 64).sum(1), [[Opt(OptOps.SPLIT, 0, (16, AxisType.UPCAST, True))]])

  def test_double_sum_group(self):
    a = Tensor.rand(4, 4, 4)
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.SPLIT, 0, (16, AxisType.LOCAL, True))],])
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 0, (16, AxisType.LOCAL, True))],])
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.SPLIT, 1, (4, AxisType.LOCAL, True)), Opt(OptOps.SPLIT, 1, (16, AxisType.LOCAL, True))],])

if __name__ == '__main__':
  unittest.main()
