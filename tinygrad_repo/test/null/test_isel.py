import unittest
from tinygrad import Tensor
from tinygrad.helpers import Target, Context
from tinygrad.codegen import to_program, pack_args
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite, ProgramInfo
from tinygrad.renderer.x86 import X86Renderer, X86Ops, IselContext

# INDEX on a register value with a constant index extracts a single element (the old GEP)
def lane(y:UOp, i:int) -> UOp: return y.index(UOp.cconst(i, dtypes.int))

class TestIselX86(unittest.TestCase):
  def isel_rewrite(self, x:UOp):
    x = pack_args(x, ProgramInfo.from_sink(x).params)
    return graph_rewrite(x, X86Renderer(Target(device="CPU", arch="x86_64")).isel_matcher, IselContext(), bottom_up=True)

  def _check_op(self, dt_op, expr):
    nargs = expr.__code__.co_argcount
    for dt,op in dt_op:
      with self.subTest(dtype=dt):
        v = [UOp.variable(str(i), 0, 0, dt) for i in range(nargs)]
        n = self.isel_rewrite(expr(*v))
        self.assertIs(n.body.arg.code, op)

  def test_cmove(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    b = UOp.variable("b", 0, 0, dtypes.int32)
    c = (a < b).where(a, b)
    d = a.alu(Ops.CMPNE, b).where(a, b)
    f = c + d
    n = self.isel_rewrite(f)
    ops = [u.body.arg.code for u in n.toposort() if u.op is Ops.CALL]
    self.assertEqual(ops.count(X86Ops.CMOVNE), 2)
    self.assertEqual(ops.count(X86Ops.SETL), 1)
    self.assertEqual(ops.count(X86Ops.SETNE), 1)
    self.assertEqual(ops.count(X86Ops.CMP), 1)

  def test_vinsertps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32)
    b = UOp.variable("b", 0, 0, dtypes.float32)
    c = UOp.variable("c", 0, 0, dtypes.float32)
    d = UOp.variable("e", 0, 0, dtypes.float32)

    valid = [UOp.stack(lane(a, 0), lane(b, 1), lane(a, 2), lane(b, 3)),
             UOp.stack(lane(a, 3), lane(b, 2), lane(c, 1), d)]
    for shuf in valid: self.assertIs(self.isel_rewrite(shuf).body.arg.code, X86Ops.VINSERTPS)

  # complex address is [base + index*scale + displacement]
  def test_complex_address(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    load = UOp.param(0, dtypes.int32, 16).index(a + UOp.cconst(1, dtypes.int32)).load()
    n = self.isel_rewrite(load)
    # displacement is the constant in "a" scaled to the buffer element size, dtype is int8 when the value fits otherwise int32
    self.assertTrue(n.src[3].dtype is dtypes.int8 and n.src[3].src[0].op is Ops.CONST and n.src[3].src[0].val == 4)

  @Context(SPEC=2, NOOPT=1)
  def test_register_store_is_effect(self):
    to_program(Tensor.empty(16, 16).sum(1).schedule_linear().src[-1].src[0], X86Renderer(Target(device="CPU", arch="x86_64")))

  def test_register_casts(self):
    renderer = X86Renderer(Target(device="CPU", arch="x86_64"))
    for t in (Tensor.empty(16, dtype=dtypes.uint32).cast(dtypes.int64), Tensor.empty(16) < Tensor.empty(16)):
      sink = to_program(t.schedule_linear().src[-1].src[0], renderer).src[0]
      for u in sink.toposort():
        if u.op is Ops.BITCAST: self.assertEqual(u.dtype.itemsize, u.src[0].dtype.itemsize)

if __name__ == "__main__":
  unittest.main()
