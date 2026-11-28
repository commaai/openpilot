import unittest
from tinygrad import Tensor, Variable, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import ParamArg, AddrSpace
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.function import function
from tinygrad.codegen import create_linear
def buf(n=8) -> Tensor: return Tensor.empty(n, device="NULL").realize()

class TestExport(unittest.TestCase):
  def test_exportable(self):
    @function
    def f(x, y): return (x*y+1).contiguous()
    v = Variable("v", 1, 8)
    multi = Tensor.empty(4, 4, device="NULL").shard(("NULL:0", "NULL:1"), axis=0).realize()
    for t in [buf()[2:6].contiguous()+1, buf().bitcast(dtypes.int32)+1, buf()[:v.bind(3)].contiguous()*v.bind(3), multi*2, f(buf(), buf()),
              Tensor.rand(4, device="NULL")]:
      fn, externs = Tensor.export(t)
      for u in fn.toposort(enter_calls=True):
        self.assertIsNot(u.op, Ops.BUFFER)
        if isinstance(u.arg, ParamArg): self.assertTrue(u.arg.buffer is None and u.arg.val is None, f"runtime state in {u}")
        if u.op is Ops.EXTERN: self.assertEqual(u.device, externs[u.arg.slot].device)
      # EXTERNs have no names, each one has a unique slot
      self.assertEqual(sorted(u.arg.slot for u in fn.toposort(enter_calls=True) if u.op is Ops.EXTERN), list(range(len(externs))))
      # UOps are hash-consed, so a lossless round trip through the wire format gives back the identical UOp
      self.assertIs(parse_uir(render_uir(fn)), fn)

  def test_symbolic_extern(self):
    v = Variable("v", 1, 8)
    fn, externs = Tensor.export(buf()[:v.bind(3)].contiguous()*v.bind(3))
    e = next(u for u in fn.toposort() if u.op is Ops.EXTERN and u.addrspace is AddrSpace.ALU)
    self.assertTrue(e.is_variable and not e.is_bound_var)
    self.assertEqual(externs[e.arg.slot].arg.val, 3)
    self.assertEqual(e.expr, f"e{e.arg.slot}")
    self.assertEqual((e*2+1).render(), f"({e.expr}*2+1)")
    self.assertEqual((e*2+1).sym_infer({e.expr: 3}), 7)
    self.assertEqual(fn.variables(), [e])
    with self.assertRaises(AssertionError): e.bind(3)

  def test_symbolic_kernel_arg(self):
    fn, _ = Tensor.export(buf()[:Variable("i", 1, 8).bind(3)] + 1)
    e, = fn.variables()
    call, = create_linear(fn).src
    p, = call.body.variables()
    self.assertIs(p.op, Ops.PARAM)
    self.assertEqual(p.arg.vmin_vmax, e.arg.vmin_vmax)
    self.assertIs(call.src[1+p.arg.slot], e)
    self.assertNotIn(e, call.body.toposort())

  def test_symbolic_program_shared_across_bindings(self):
    from tinygrad.codegen.compile import lower_and_compile
    def compile_slice(name, value):
      linear, externs = (buf()[:Variable(name, 1, 8).bind(value)] + 1).linear_with_externs()
      call, = linear.src
      p, = call.body.variables()
      self.assertIsNone(p.arg.name)
      self.assertEqual(externs[call.src[1+p.arg.slot].arg.slot].unbind(), (Variable(name, 1, 8), value))
      return lower_and_compile(linear).src[0].body
    self.assertIs(compile_slice("i", 2), compile_slice("j", 5))

  def test_same_function(self):
    v = Variable("v", 1, 4)
    a, b, c, d = buf(), buf(), buf(), buf()
    fn, _ = Tensor.export((a+b).contiguous()*v.bind(2))
    self.assertIs(fn, Tensor.export((c+d).contiguous()*v.bind(3))[0])
    # its externs are this storage and these values
    _, externs = ((c+d).contiguous()*v.bind(3)).linear_with_externs()
    self.assertEqual({x.expr: x.arg.val for x in externs if x.is_bound_var}, {"v": 3})
    self.assertTrue({c.uop.buf_uop, d.uop.buf_uop} <= set(externs))

  def test_create_linear_deterministic(self):
    # the linear only depends on the export, the storage it allocates gets canonical slots
    from tinygrad.codegen import create_linear, schedule_cache
    fn, _ = Tensor.export((((buf()+1).contiguous()*2).to("NULL:1")+3).contiguous())
    linear = create_linear(fn)
    self.assertTrue(any(u.op is Ops.ALLOC for u in linear.toposort()))
    Tensor.empty(4, device="NULL").realize()  # other work takes slots from the global counter
    self.assertIs(create_linear(fn), linear)
    schedule_cache.clear()
    self.assertIs(create_linear(fn), linear)

  def test_create_linear_cached(self):
    # the linear of a function is cached for as long as its schedule is
    from unittest.mock import patch
    from tinygrad import Context
    import tinygrad.codegen as codegen
    @function(precompile=True)
    def add_one(x): return (x+1).contiguous()
    fn, _ = Tensor.export((add_one(buf())*2+3).contiguous())
    codegen.schedule_cache.clear()
    with patch("tinygrad.codegen.memory_plan_rewrite", wraps=codegen.memory_plan_rewrite) as plan:
      linear = create_linear(fn)
      self.assertIs(create_linear(fn), linear)
      self.assertEqual(plan.call_count, 1)
      # the jit defers the memory plan, a planned linear is not what it gets
      with Context(NO_MEMORY_PLANNER=1): self.assertIsNot(create_linear(fn), linear)
      with Context(SCACHE=0): self.assertIs(create_linear(fn), linear)
      self.assertEqual(plan.call_count, 3)
      codegen.schedule_cache.clear()
      self.assertIs(create_linear(fn), linear)
      self.assertEqual(plan.call_count, 4)

  def test_linear_cache_isolation(self):
    from tinygrad import Context
    from tinygrad.codegen import schedule_cache
    fn, _ = Tensor.export(buf(65536).sum())
    schedule_cache.clear()
    with Context(SCACHE=1, SPLIT_REDUCEOP=0, NO_MEMORY_PLANNER=0):
      linear = create_linear(fn)
      self.assertEqual(len(linear.src), 1)
      with Context(SCACHE=0, SPLIT_REDUCEOP=1): self.assertEqual(len(create_linear(fn).src), 2)
      self.assertIs(create_linear(fn), linear)  # disabled cache didn't overwrite the planned entry
      schedule_cache.clear()
      with Context(NO_MEMORY_PLANNER=1, SPLIT_REDUCEOP=1): self.assertEqual(len(create_linear(fn).src), 2)
      self.assertEqual(len(create_linear(fn).src), 2)  # use the rebuilt schedule, not the old planned entry

if __name__ == "__main__":
  unittest.main()
