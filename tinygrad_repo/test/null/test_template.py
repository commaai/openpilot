import unittest
from tinygrad import Context, Device, Tensor, UOp, dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.dtype import AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import KernelInfo, graph_rewrite, pm_resolve_params
from tinygrad.uop.spec import spec_kernel_graph, spec_program, spec_tensor, type_verify
from tinygrad.uop.uir import parse_uir, render_uir

class TestTemplateParams(unittest.TestCase):
  def test_scalar_and_shapes(self):
    self.assertFalse(hasattr(Ops, "TEMPLATE"))
    with Context(SPEC=0):
      n, m = UOp.template("N", (1, 128)), UOp.template("M", (0, 64), dtypes.uint32)
      eps, scale = UOp.template("eps", (0.0, 1.0), dtypes.float32), UOp.template("s", dtype=dtypes.float16)
      self.assertEqual((n.op, n.addrspace, n.arg.slot, n.arg.name, n.shape), (Ops.PARAM, AddrSpace.TEMPLATE, -1, "N", ()))
      self.assertEqual(((n + 2) * 3).specialize(N=4), UOp.const(18))
      self.assertEqual((m + 1).specialize(M=7), UOp.const(8, dtypes.uint32))
      self.assertEqual((eps * 2.0 + 1.0).specialize(eps=0.25), UOp.const(1.5, dtypes.float32))
      self.assertEqual((scale * 2.0).specialize(s=1.5), UOp.const(3.0, dtypes.float16))
      fixed = UOp.template("K", (4, 4))
      self.assertIs(fixed.simplify(), fixed)
      self.assertEqual((fixed + 1).specialize(K=4), UOp.const(5))
      d, u, v = UOp.template("D", (32, 4096)), UOp.template("U"), UOp.variable("v", 1, 16)
      p1, p2 = UOp.param(0, dtypes.float32, (n,)), UOp.param(1, dtypes.float16, (n, d))
      a = UOp.alloc((u, d // 32, 8), dtypes.uint32, slot=7, device="NULL")
      self.assertEqual((p1.shape, p2.shape, a.shape), ((n,), (n, d), (u, d // 32, 8)))
      self.assertIs(p1.specialize(N=12), UOp.param(0, dtypes.float32, (12,)))
      self.assertIs(p2.specialize(N=4, D=128), UOp.param(1, dtypes.float16, (4, 128)))
      self.assertIs(a.specialize(U=2, D=256), UOp.alloc((2, 8, 8), dtypes.uint32, slot=7, device="NULL"))
      pmix = UOp.param(0, dtypes.float32, (v, d))
      self.assertEqual(pmix.variables(), [v])
      self.assertIs(pmix.specialize(D=64), UOp.param(0, dtypes.float32, (v, 64)))
      self.assertFalse(n.is_variable or n.is_bound_var)
      self.assertEqual((n + v).variables(), [v])
      self.assertIs(graph_rewrite(n + v, pm_resolve_params, ctx=(UOp.const(99),), walk=True), n + v)

  def test_uir_roundtrip_and_spec_enforcement(self):
    with Context(SPEC=0):
      n, d, eps = UOp.template("N", (1, 128)), UOp.template("D"), UOp.template("eps", (0.0, 1.0), dtypes.float32)
      out, inp = UOp.param(0, dtypes.float32, (n, d)), UOp.param(1, dtypes.float32, (n, d))
      r0, r1 = UOp.range(n, 0), UOp.range(d, 1)
      kern = out[r0, r1].store(inp[r0, r1].load() + eps).end(r0, r1).sink(arg=KernelInfo(name="add_eps", opts_to_apply=()))
      txt = render_uir(kern, color=False)
      self.assertIn("addrspace=TEMPLATE", txt)
      self.assertNotIn("\x1b", txt)
      self.assertIs(parse_uir(txt), kern)
      spec_k = parse_uir(txt).specialize(N=2, D=4, eps=0.5)
      type_verify(spec_k, spec_tensor)
      for s in (spec_tensor, spec_program, spec_kernel_graph):
        with self.assertRaises(RuntimeError): type_verify(kern, s)
    with Context(SPEC=2):
      with self.assertRaises(RuntimeError): full_rewrite_to_sink(kern, Device["CPU"].renderer, optimize=False)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device="CPU").realize()
    y = Tensor.empty(2, 4, dtype=dtypes.float32, device="CPU")
    self.assertEqual(Tensor(y.uop.after(spec_k.call(y.uop, x.uop))).tolist(), [[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5]])

  def test_validation(self):
    with Context(SPEC=0):
      n, eps = UOp.template("N", (1, 16)), UOp.template("eps", (0.0, 1.0), dtypes.float32)
      g = n.cast(dtypes.float32) + eps
      for kw in ({"N": 4}, {"N": 4, "eps": 0.5, "extra": 1}, {"N": True, "eps": 0.5},
                 {"N": 0, "eps": 0.5}, {"N": 17, "eps": 0.5}, {"N": 4, "eps": -0.1}, {"N": 4, "eps": float("nan")}):
        with self.assertRaises(AssertionError): g.specialize(**kw)  # type: ignore
      self.assertIs(UOp.const(5).specialize(), UOp.const(5))
      with self.assertRaises(AssertionError): UOp.const(5).specialize(N=1)

if __name__ == "__main__":
  unittest.main()
