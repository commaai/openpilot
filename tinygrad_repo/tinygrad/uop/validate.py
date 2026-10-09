from typing import Callable
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, python_alu, range_str
from tinygrad.dtype import dtypes, Invalid
from tinygrad.helpers import cpu_profile
import z3

# older versions of z3 dont have some operators like & overloaded
if z3.get_version() < (4, 12, 4, 0):
  raise ImportError("bounds checking requires z3 >= 4.12.4, use CHECK_OOB=0 to disable, or \"pip install 'z3-solver>=4.12.4\"")

# IDIV is truncated division but z3 does euclidian division (floor if b>0 ceil otherwise); mod by power of two sometimes uses Ops.AND
def z3_cdiv(a:z3.ArithRef, b:z3.ArithRef) -> z3.ArithRef:return z3.If((a<0), z3.If(0<b, (a+(b-1))/b, (a-(b+1))/b), a/b)
def z3_floordiv(a:z3.ArithRef, b:z3.ArithRef) -> z3.ArithRef: return z3.If(b > 0, a/b, (-a)/(-b))
# general int bitops are bit-blasted through the narrowest BV that fits both sides in two's complement (BV ops get exponentially slower with width)
def z3_bv(x:UOp, op:Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef], a:z3.ExprRef, b:z3.ExprRef) -> z3.ExprRef:
  w = 1 + max(int(x.src[0].vmax).bit_length(), int(x.src[1].vmax).bit_length(), int(-x.src[0].vmin).bit_length(), int(-x.src[1].vmin).bit_length())
  return z3.BV2Int(op(z3.Int2BV(a, w), z3.Int2BV(b, w)), is_signed=True)
def z3_xor(x:UOp, a:z3.ExprRef, b:z3.ExprRef) -> z3.ExprRef:
  if isinstance(a, z3.BoolRef): return a^b
  # x ^ -1 = -(x+1), i.e. bitwise NOT
  if isinstance(b, z3.IntNumRef) and b.as_long() == -1: return -(a+1)
  if isinstance(a, z3.IntNumRef) and a.as_long() == -1: return -(b+1)
  return z3_bv(x, lambda u,v: u^v, a, b)
def z3_and(x:UOp, a:z3.ExprRef, b:z3.ExprRef) -> z3.ExprRef:
  if isinstance(a, z3.BoolRef): return a&b
  if isinstance(a, z3.IntNumRef): a, b = b, a
  if isinstance(b, z3.IntNumRef):
    # x & (2^k-1) = x % 2^k and x & -(2^k) = x - x % 2^k for any x in two's complement
    if (m:=b.as_long()+1) > 0 and m&(m-1) == 0: return a%m
    if (m:=-b.as_long()) > 0 and m&(m-1) == 0: return a - a%m
  return z3_bv(x, lambda u,v: u&v, a, b)
def z3_or(x:UOp, a:z3.ExprRef, b:z3.ExprRef) -> z3.ExprRef:
  return a|b if isinstance(a, z3.BoolRef) else z3_bv(x, lambda u,v: u|v, a, b)

# Factor out the minimum count, then shift by its varying bits. Constant counts need no stages.
def z3_shift(x:UOp, a:z3.ExprRef, b:z3.ExprRef) -> z3.ExprRef:
  lo = max(0, int(x.src[1].vmin))
  a = a / (1 << lo) if x.op is Ops.SHR else a * (1 << lo)
  for i in range(max(0, int(x.src[1].vmax)-lo).bit_length()):
    factor = 1 << (1 << i)
    a = z3.If(((b-lo) / (1 << i)) % 2 == 1, a / factor if x.op is Ops.SHR else a * factor, a)
  return z3.If(b < 0, z3.FreshInt("invalid_shift", ctx=a.ctx), a)

# handlers take (x, *src exprs): x provides vmin/vmax for the bounds-aware ops; anything not here falls back to python_alu
z3_alu: dict[Ops, Callable[..., z3.ExprRef]] = {Ops.CMOD: lambda _,a,b: a-z3_cdiv(a,b)*b, Ops.CDIV: lambda _,a,b: z3_cdiv(a,b),
  Ops.FLOORDIV: lambda _,a,b: z3_floordiv(a,b), Ops.FLOORMOD: lambda _,a,b: a-z3_floordiv(a,b)*b,
  Ops.WHERE: lambda _,c,a,b: z3.If(c,a,b), Ops.MAX: lambda _,a,b: z3.If(a<b, b, a),
  Ops.AND: z3_and, Ops.OR: z3_or, Ops.XOR: z3_xor, Ops.SHL: z3_shift, Ops.SHR: z3_shift}

def create_bounded(name:str, vmin:int|z3.ArithRef, vmax:int|z3.ArithRef, solver:z3.Solver) -> z3.ArithRef:
  solver.add((vmin <= (s:=z3.Int(name, ctx=solver.ctx)))&(s <= vmax))
  return s
def create_var(x:UOp, ctx:tuple[z3.Solver, dict[UOp, z3.ExprRef]]) -> z3.ExprRef:
  name = x.arg.name if x.op is Ops.PARAM else f"{x.op.name.lower()}{len(ctx[1])}"
  return z3.Bool(name, ctx=ctx[0].ctx) if x.dtype == dtypes.bool else create_bounded(name, x.vmin, x.vmax, ctx[0])
# z3 does not model widths: a cast only converts between bool and int
def z3_cast(c:UOp, x:z3.ExprRef) -> z3.ExprRef:
  if (c.src[0].dtype == dtypes.bool) == (c.dtype == dtypes.bool): return x
  return x != 0 if c.dtype == dtypes.bool else z3.If(x, 1, 0)

z3_renderer = PatternMatcher([
  # the valid condition is a constraint
  (UPat.var("cond").where(UPat.var("x"), UPat(Ops.CONST, arg=Invalid)), lambda x,cond,ctx: ctx[0].add(ctx[1][cond]) or ctx[1][x]),
  # variables
  (UPat((Ops.SPECIAL, Ops.RANGE), name="x"), lambda x,ctx:
   create_bounded(x.arg if x.op is Ops.SPECIAL else f"r{range_str(x)}", 0, ctx[1][x.src[0]]-1, ctx[0])),
  # unknown values are variables bounded by their vmin/vmax: params, loads (non-pointer INDEX is a LOAD) and anything from floats
  # an AFTER passes its value through, its extra srcs are ordering deps and are ignored
  (UPat(Ops.AFTER, name="x"), lambda x,ctx: ctx[1][x.src[0]] if x.src[0] in ctx[1] else create_var(x, ctx)),
  (UPat((Ops.PARAM, Ops.LOAD, Ops.INDEX), name="x"), create_var),
  (UPat((Ops.CAST, Ops.BITCAST)+tuple(GroupOp.Comparison), src=UPat(dtype=dtypes.floats), name="x"), create_var),
  # a bitcast between ints wraps into the target range, z3 ints are unbounded
  (UPat(Ops.BITCAST, dtypes.ints, src=(UPat.var("x", dtypes.ints),), name="c"),
   lambda c,x,ctx: (ctx[1][x]-c.dtype.min) % 2**(8*c.dtype.itemsize) + c.dtype.min),
  # constants
  (UPat(Ops.CONST, arg=Invalid), lambda ctx: z3.Int("Invalid", ctx=ctx[0].ctx)),
  (UPat(Ops.CONST, name="x"), lambda x,ctx: z3.BoolVal(x.val, ctx=ctx[0].ctx) if x.dtype == dtypes.bool else z3.IntVal(x.val, ctx=ctx[0].ctx)),
  (UPat(Ops.CAST, src=(UPat.var("x"),), name="c"), lambda c,x,ctx: z3_cast(c, ctx[1][x])),
  (UPat(tuple(z3_alu), name="x"), lambda x,ctx: z3_alu[x.op](x, *(ctx[1][s] for s in x.src))),
  (UPat(GroupOp.ALU, name="x"), lambda x,ctx: python_alu[x.op](*(ctx[1][s] for s in x.src))),
])

def uops_to_z3(solver:z3.Solver, *uops: UOp) -> list[z3.ExprRef]:
  # gate on upstream memory addressing, but keep INDEX as an unknown LOAD
  lst = list(UOp.sink(*uops).toposort(gate=lambda x: x.op not in {Ops.SHRINK, Ops.ALLOC, Ops.BUFFER}
                                      and (x.dtype in dtypes.ints+(dtypes.bool, dtypes.weakint) or x.op is Ops.SINK)))[:-1]
  z3map: dict[UOp, z3.ExprRef] = {}
  for u in lst:
    if (z3_rewritten:=z3_renderer.rewrite(u, ctx=(solver, z3map))) is None: raise NotImplementedError(f"{u.op} is not supported by z3")
    z3map[u] = z3_rewritten
  assert all(u in z3map for u in uops), "UOp failed to rewrite to z3!"
  return [z3map[u] for u in uops]

def validate_index_with_z3(sz:int, idx:UOp, gate:UOp) -> bool:
  solver = z3.Solver(ctx=z3.Context())
  z3_idx, z3_mask = uops_to_z3(solver, idx, gate)
  solver.add(z3_mask)
  with cpu_profile("validate index with z3", "TINY"):
    match solver.check((z3_idx<0)|(sz<=z3_idx)):
      case z3.unsat: return True
      case z3.sat: print(f"# OUT OF BOUNDS ACCESS: at {solver.model()} INDEX not in 0 - {sz}\nconstraints = {solver}")
      case z3.unknown: print(f"# UNKNOWN RESULT FROM Z3: {solver.reason_unknown()}\nconstraints = {solver}")
  print(f"idx={idx.render(simplify=False)}")
  print(f"mask={gate.render(simplify=False)}")
  return False
