import unittest
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import Ops, GroupOp, AxisType
from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.codegen import to_program
from tinygrad.dtype import DType, dtypes, AddrSpace
from tinygrad.renderer.isa import ISARenderer
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.helpers import Target
from test.helpers import replace_opts

@unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, ISARenderer), "isa backends don't preserve the op spec when lowering")
class TestLinearizer(unittest.TestCase):
  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]

    uops = tuple(to_program(replace_opts(r.schedule_linear().src[-1].src[0], [Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST))]),
                       renderer=Device[Device.DEFAULT].renderer).src[1].src)
    num_loads = len([uop for uop in uops if uop.op is Ops.LOAD])
    assert num_loads <= 4, "more load uops than needed"
    assert num_loads >= 1, "expected at least one load uop"

  @unittest.skip("this is handled at higher level now")
  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])

    uops = tuple(to_program(replace_opts(r.schedule_linear().src[-1].src[0], [Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST))]),
                       renderer=Device[Device.DEFAULT].renderer).src[1].src)
    num_ops = len([uop for uop in uops if uop.op in GroupOp.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_reduce_upcast(self):
    x, w = Tensor.randn((1,1,3)).realize(), Tensor.randn((1,1,2)).realize()
    r = Tensor.conv2d(x,w,padding=1).relu()

    uops = tuple(to_program(replace_opts(r.schedule_linear().src[-1].src[0],
      [Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)),
       Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))]), renderer=Device[Device.DEFAULT].renderer).src[1].src)
    accs = [u for u in uops if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG]
    stores = [u for u in uops if u.op is Ops.STORE]
    assert len(accs) == 0  # it's removed now
    assert len(stores) == 1

  def test_zero_fold(self):
    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack(a, b)
    uops = tuple(to_program(replace_opts(r.schedule_linear().src[-1].src[0], [Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST))]),
                       renderer=Device[Device.DEFAULT].renderer).src[1].src)
    num_ops = len([uop for uop in uops if uop.op in GroupOp.ALU])
    assert num_ops == 0, "more alu uops than needed"

  def test_sum_acc_dtype(self):
    for tensor_dtype, acc_dtype in (
      (dtypes.bool, dtypes.int), (dtypes.int16, dtypes.int), (dtypes.float16, dtypes.float), (dtypes.bfloat16, dtypes.float)):
      if tensor_dtype in (dts:=Device[Device.DEFAULT].renderer.supported_dtypes()) and acc_dtype in dts:
        a = Tensor([1, 2, 3], dtype=tensor_dtype).sum()
        realized_ast = a.schedule_linear().src[-1].src[0]
        program = to_program(replace_opts(realized_ast, []), renderer=Device[Device.DEFAULT].renderer)
        local = [uop for uop in tuple(program.src[1].src) if uop.op is Ops.BUFFER and uop.addrspace in (AddrSpace.LOCAL, AddrSpace.REG)]
        assert local[0].dtype == acc_dtype

  def test_arg_acc_dtype(self):
    def helper_arg_acc_dtype(c: Tensor, expected_dtype:DType):
      realized_ast = c.schedule_linear().src[-1].src[0]
      program = to_program(replace_opts(realized_ast, []), renderer=Device[Device.DEFAULT].renderer)
      local = [uop for uop in tuple(program.src[1].src) if uop.op is Ops.BUFFER and uop.addrspace in (AddrSpace.LOCAL, AddrSpace.REG)]
      self.assertEqual(local[0].dtype, expected_dtype)

    tests = (
      (dtypes.float16, None, dtypes.float),
      (dtypes.bfloat16, None, dtypes.float),
      (dtypes.float, None, dtypes.float),
      (dtypes.float16, dtypes.float16, dtypes.float16),
      (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16),
      (dtypes.float, dtypes.float16, dtypes.float16),
    )
    for tensor_dtype, acc_dtype, expected_dtype in tests:
      if tensor_dtype in (dts:=Device[Device.DEFAULT].renderer.supported_dtypes()) and acc_dtype in dts|{None} and expected_dtype in dts:
        a, b = Tensor.rand(8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, dtype=tensor_dtype)
        helper_arg_acc_dtype(a.sum(dtype=acc_dtype), expected_dtype)
        helper_arg_acc_dtype(a.matmul(b, dtype=acc_dtype), expected_dtype)
        helper_arg_acc_dtype(Tensor.einsum("ki,ij->kj", a, b, dtype=acc_dtype), expected_dtype)
        d, w = Tensor.rand(4, 8, 8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, 2, 2, dtype=tensor_dtype)
        helper_arg_acc_dtype(d.conv2d(w, dtype=acc_dtype), expected_dtype)

  def test_sum_collapse(self):
    t = Tensor([2]).reshape(1, 1).expand(256, 256).sum()
    sched = [si for si in t.schedule_linear().src if si.src[0].op is Ops.SINK]
    # sum_collapse is a full collapse now
    assert len(sched) == 1
    assert not any(u.op is Ops.REDUCE and u.arg[1] > 0 for u in sched[0].src[0].toposort()), "found reduce in sum collapse"

class TestLinearizerRenderers(unittest.TestCase):
  # NOTE: can reenable, it does work. it just makes BEAM slow
  @unittest.expectedFailure
  def test_upcast_with_locals_cpu(self):
    out = Tensor.ones(64,64).contiguous() @ Tensor.ones(64,64).contiguous()
    prg = to_program(replace_opts(out.schedule_linear().src[-1].src[0], [Opt(OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL))]),
                      renderer=ClangRenderer(Target("CPU", arch="x86_64,x86-64")))
    self.assertEqual(len(prg.src[2].arg.split("for")), 5)

  def test_upcast_with_locals(self):
    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    opts_to_apply = [Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)),
                     Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))]
    program = to_program(replace_opts(r.schedule_linear().src[-1].src[0], opts_to_apply),
                         renderer=AMDLLVMRenderer(Target("AMD", arch="gfx1100")))

    stores = [u for u in tuple(program.src[1].src) if u.op is Ops.STORE and u.src[0].addrspace != AddrSpace.REG]

    # the first store is to lds and can be upcasted
    assert stores[0].src[1].max_numel() == 4
    assert any(x.addrspace is AddrSpace.LOCAL for x in stores[0].toposort())
    # the second store is to gds with no upcasts
    assert stores[1].src[1].max_numel() == 1
    assert stores[1].src[1].dtype == dtypes.float
    assert any(x.op is Ops.PARAM for x in stores[1].toposort())

if __name__ == '__main__':
  unittest.main()
