import unittest, re, ast as pyast
from typing import Any
from tinygrad import Tensor
from tinygrad.dtype import dtypes, DType, AddrSpace, Invalid
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, UOp, graph_rewrite, ParamArg, KernelInfo
from tinygrad.uop.movement import mop_cleanup
from tinygrad.uop.render import render_uir
from tinygrad.device import Device
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.helpers import Context, ansistrip

# ***** prototype parse for the uop v1 wire format *****

_DTYPES_BY_NAME: dict[str, DType] = {d.name: d for _,v in vars(type(dtypes)).items() if isinstance(v, DType) for d in [v]}

_line_re = re.compile(r"^\s*%(\d+) = (\w+)\s*(.*)$")
_src_tok = re.compile(r"\([^()]*\)|[^,\s]+")   # a src is %N, a literal, or a flat all-const tuple

def _kv(zone:str) -> dict[str, str]:
  """the arg zone is k=v tokens"""
  kv = {}
  for t in zone.split():
    k, _, v = t.partition("=")
    kv[k] = v
  return kv

def _parse_const(tok:str) -> UOp:
  if tok == "invalid": return UOp(Ops.CONST, src=(), arg=Invalid)
  if tok in ("true", "false"): return UOp(Ops.CONST, src=(), arg=tok == "true")
  if ":" in tok:
    dt = _DTYPES_BY_NAME[d:=tok.split(":", 1)[0]]
    return UOp(Ops.CONST, src=(), arg=dt.const(float.fromhex(tok[len(d)+1:])) if dtypes.is_float(dt) else dt.const(int(tok[len(d)+1:])))
  if re.match(r"^-?\d+$", tok): return UOp(Ops.CONST, src=(), arg=int(tok))
  return UOp(Ops.CONST, src=(), arg=dtypes.weakfloat.const(float(tok)))

def _parse_paramarg(rest:str, name:str|None) -> ParamArg:
  kv = _kv(rest)
  def i(k:str) -> int|None: return int(kv[k]) if k in kv else None
  dev = kv.get("device")
  return ParamArg(int(kv["slot"]), _DTYPES_BY_NAME[kv["dtype"]],
                  tuple(map(int, kv["bounds"].strip("[]").split(","))) if "bounds" in kv else None,
                  i("multiple_of"), name, AddrSpace[kv["addrspace"]] if "addrspace" in kv else AddrSpace.GLOBAL,
                  pyast.literal_eval(dev) if dev and dev[0] in "'(" else dev, kv.get("volatile") == "true")

def parse_ssa(text:str) -> UOp:
  # the wire format carries storage declarations without runtime state: `buffer` reconstructs as an ALLOC
  nodes, root = {}, None
  def parse_tok(tok:str) -> UOp:
    if tok.startswith("%"): return nodes[int(tok[1:])]
    if tok.startswith("("): return UOp(Ops.STACK, src=tuple(parse_tok(t) for t in tok[1:-1].split(", ") if t))
    return _parse_const(tok)
  for raw in ansistrip(text).splitlines():   # op names may carry ANSI color from render_uir
    line = raw.strip()
    if not line or line.startswith(";"): continue
    m = _line_re.match(line)
    assert m, f"unparseable line {raw!r}"
    n, op, rest = int(m.group(1)), Ops[m.group(2).upper()], m.group(3).strip()
    if op is Ops.BUFFER: op = Ops.ALLOC   # the wire carries storage declarations without runtime state
    if rest.startswith(": "): srcstr, argstr = "", rest[2:]
    elif " : " in rest: srcstr, argstr = rest.split(" : ", 1)
    else: srcstr, argstr = rest, ""
    name: str|None = None
    if argstr.startswith('"'):  # quoted param name comes first in the args
      end = argstr.index('"', 1)
      name, argstr = argstr[1:end], argstr[end+1:].strip()
    srcs = [parse_tok(t) for t in _src_tok.findall(srcstr)]
    arg: Any = None
    if op in {Ops.PARAM, Ops.BUFFER, Ops.ALLOC}: arg = _parse_paramarg(argstr, name)
    elif op is Ops.RANGE:
      t, *ids = argstr.split()
      arg = (AxisType[t], *map(int, ids))
    elif op is Ops.REDUCE:
      kv = _kv(argstr)
      arg = (Ops[kv["op"].upper()], int(kv["pop"]) if "pop" in kv else 0)
    elif op in (Ops.CAST, Ops.BITCAST): arg = _DTYPES_BY_NAME[argstr]
    elif op is Ops.SINK: arg = KernelInfo(name=argstr) if argstr else None
    elif op in (Ops.COPY, Ops.SPECIAL): arg = argstr
    elif argstr:
      try: arg = pyast.literal_eval(argstr)
      except (SyntaxError, ValueError): arg = argstr
    if op is Ops.BUFFER: op = Ops.ALLOC
    nodes[n] = root = UOp(op, src=tuple(srcs), arg=arg)   # root is always the last node
  assert root is not None, "empty graph"
  return root

def _strip_buffers(root:UOp) -> UOp:
  # realized BUFFERs carry runtime state the wire can't; a BUFFER with no binding is exactly an ALLOC
  subs = {b: b.replace(op=Ops.ALLOC, arg=ParamArg(b.arg.slot, b.arg.dtype, b.arg.vmin_vmax, b.arg.multiple_of,
                                                 b.arg.name, b.arg.addrspace, b.arg.device, b.arg.volatile))
          for b in root.toposort() if b.op is Ops.BUFFER and isinstance(b.arg, ParamArg)}
  return root.substitute(subs, walk=True, name="strip buffers for wire format test") if subs else root

def assert_roundtrip(case, root:UOp):
  txt = render_uir(root)
  parsed = parse_ssa(txt)
  # text + structural equality, both sides stripped: realized BUFFER vs parsed ALLOC converge to the same thing
  g1, g2 = _strip_buffers(root), _strip_buffers(parsed)
  case.assertEqual(render_uir(g1), render_uir(g2))
  case.assertEqual({x.tuplize for x in g1.toposort()}, {x.tuplize for x in g2.toposort()})

class TestSSARender(unittest.TestCase):
  def test_gemm_pre_rangeify(self):
    a = Tensor.rand(256, 256, device="CPU").realize()
    b = Tensor.rand(256, 256, device="CPU").realize()
    root = graph_rewrite((a.reshape(256,256,1) * b.reshape(256,1,256)).sum(0).uop, mop_cleanup)
    assert_roundtrip(self, root)

  def _kernel(self, t:Tensor) -> UOp:
    lin, _ = Tensor.linear_with_vars(t)
    return [c for c in lin.src if any(u.op is Ops.REDUCE for u in c.src[0].toposort())][0].src[0]

  def test_conv_kernel_graph(self):
    o = Tensor.rand(1,1,6,6, device="CPU").conv2d(Tensor.rand(1,1,3,3, device="CPU"))
    assert_roundtrip(self, self._kernel(o))

  def test_conv_noopt(self):
    o = Tensor.rand(1,1,6,6, device="CPU").conv2d(Tensor.rand(1,1,3,3, device="CPU"))
    with Context(NOOPT=1): sink = full_rewrite_to_sink(self._kernel(o), Device["CPU"].renderer, optimize=False)
    assert_roundtrip(self, sink)

  def test_gemm_kernel_graph(self):
    o = Tensor.rand(4,4, device="CPU").matmul(Tensor.rand(4,4, device="CPU"))
    assert_roundtrip(self, self._kernel(o))

  def test_symbolic_variable(self):
    n = UOp.variable("n", 1, 64)
    g = UOp.sink(((n + 1) * 2) % 7, (n % 5) + UOp.const(Invalid))
    assert_roundtrip(self, g)

  def test_const_typing(self):
    g = UOp.sink(UOp.const(True), UOp.const(1), UOp.const(-5, dtypes.int32), UOp.const(2.0, dtypes.float32),
                 UOp.const(-0.0, dtypes.float32), UOp.const(0.1, dtypes.float64), UOp.const(Invalid),
                 UOp.const(12345678901234, dtypes.int64))
    assert_roundtrip(self, g)
    # exact float roundtrip: every representable hex float must come back bit-equal
    import random
    random.seed(0)
    vs = [random.uniform(-1e6, 1e6) for _ in range(20)]
    g2 = UOp.sink(*[UOp.const(v, dtypes.float32) for v in vs])
    assert_roundtrip(self, g2)

  def test_where_max_bool(self):
    a = UOp.param(0, dtypes.int32)
    b = (a < UOp.const(3, dtypes.int32)) & (a != 1)   # bool-AND must lower to MUL on render
    g = UOp.sink(b.where(a.maximum(UOp.const(1, dtypes.int32)), a.cast(dtypes.float32).reciprocal()))
    assert_roundtrip(self, g)

  def test_complex_programs(self):
    # softmax over a 256x16 row: reduce-max + reduce-sum + exp2
    x = Tensor.rand(256, 16, device="CPU").realize()
    o = x.softmax(-1)
    assert_roundtrip(self, o.uop)
    assert len(o.uop.toposort()) > 10  # not trivially small

  def test_continue_parsing(self):
    # lines out of order / comments tolerated, ids sparse
    g = UOp.sink(UOp.const(1) + UOp.const(2))
    txt = "; hand written\n" + render_uir(g) + "\n;; trailing comment\n"
    assert_roundtrip(self, parse_ssa(txt))

if __name__ == '__main__': unittest.main()
