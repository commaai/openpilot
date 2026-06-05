from typing import Callable, cast
from tinygrad.uop.ops import PatternMatcher, UPat, GroupOp, Ops, UOp, python_alu
from tinygrad.dtype import dtypes, Invalid
from tinygrad.helpers import cpu_profile
import z3

# older versions of z3 dont have some operators like & overloaded
if z3.get_version() < (4, 12, 4, 0):
  raise ImportError("bounds checking requires z3 >= 4.12.4, use CHECK_OOB=0 to disable, or \"pip install 'z3-solver>=4.12.4\"")

# IDIV is truncated division but z3 does euclidian division (floor if b>0 ceil otherwise); mod by power of two sometimes uses Ops.AND
def z3_cdiv(a:z3.ArithRef, b:z3.ArithRef) -> z3.ArithRef:return z3.If((a<0), z3.If(0<b, (a+(b-1))/b, (a-(b+1))/b), a/b)
def z3_floordiv(a:z3.ArithRef, b:z3.ArithRef) -> z3.ArithRef: return z3.If(b > 0, a/b, (-a)/(-b))
def z3_xor(a:z3.ExprRef, b:z3.ExprRef) -> z3.ExprRef:
  if isinstance(a, z3.BoolRef): return a^b
  # x ^ -1 = -(x+1), i.e. bitwise NOT
  if isinstance(b, z3.IntNumRef) and b.as_long() == -1: return -(a+1)
  if isinstance(a, z3.IntNumRef) and a.as_long() == -1: return -(b+1)
  raise RuntimeError(f"z3 int XOR only supports XOR with -1, got {a=} {b=}")
z3_alu: dict[Ops, Callable[..., z3.ExprRef]] = python_alu | {Ops.CMOD: lambda a,b: a-z3_cdiv(a,b)*b, Ops.CDIV: z3_cdiv, Ops.FLOORDIV: z3_floordiv,
  Ops.FLOORMOD: lambda a,b: a-z3_floordiv(a,b)*b,
  Ops.SHR: lambda a,b: a/(2**b.as_long()), Ops.SHL: lambda a,b: a*(2**b.as_long()),
  Ops.AND: lambda a,b: a%(b+1) if isinstance(b, z3.ArithRef) else a&b, Ops.WHERE: z3.If, Ops.XOR: z3_xor, Ops.MAX: lambda a,b: z3.If(a<b, b, a),}
def create_bounded(name:str, vmin:int, vmax:int, z3ctx:z3.Context) -> tuple[z3.ArithRef, z3.BoolRef]:
  return (s:=z3.Int(name, ctx=z3ctx)), (vmin <= s)&(s <= vmax)

z3_renderer = PatternMatcher([
  (UPat.var("cond").where(UPat.var("x"), UPat.const(dtypes.weakint, Invalid)), lambda x,cond,ctx: (ctx[1][x], ctx[1][cond])),
  # variables
  (UPat(Ops.SPECIAL, name="x"), lambda x,ctx: create_bounded(x.arg, 0, ctx[1][x.src[0]]-1, ctx[0])),
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x,ctx: create_bounded(x.arg[0], x.arg[1], x.arg[2], ctx[0])),
  (UPat(Ops.RANGE, name="x"), lambda x,ctx: create_bounded(x.render(simplify=False), 0, ctx[1][x.src[0]]-1, ctx[0])),
  # loads are variables bounded by the min/max of the dtype. non-pointer INDEX is also a LOAD
  (UPat((Ops.LOAD, Ops.INDEX), dtypes.ints+(dtypes.weakint,), name="x"), lambda x,ctx:
    create_bounded(f"load{len(ctx[1])}", x.dtype.min, x.dtype.max, ctx[0])),
  (UPat((Ops.LOAD, Ops.INDEX), dtypes.bool), lambda ctx: (z3.Bool(f"load{len(ctx[1])}", ctx=ctx[0]), None)),
  # constants
  (UPat(Ops.CONST, arg=Invalid), lambda ctx: (z3.Int("Invalid", ctx=ctx[0]), None)),
  (UPat(Ops.CONST, dtypes.ints+(dtypes.weakint,), name="x"), lambda x,ctx: (z3.IntVal(x.arg, ctx=ctx[0]), None)),
  (UPat(Ops.CONST, dtypes.bool, name="x"), lambda x,ctx: (z3.BoolVal(x.arg, ctx=ctx[0]), None)),
  # casts from floats create new variables
  (UPat(Ops.CAST, dtypes.ints+(dtypes.weakint,), src=(UPat(dtype=dtypes.floats),), name="x"), lambda x,ctx:
    create_bounded(f"cast{len(ctx[1])}", x.dtype.min, x.dtype.max, ctx[0])),
  # A comparison between floats introduces a new bool variable
  (UPat(GroupOp.Comparison, src=UPat(dtype=dtypes.floats)), lambda ctx: (z3.Bool(f"float_cmp{len(ctx[1])}", ctx=ctx[0]), None)),
  # casts from bool/int to int/bool
  (UPat(Ops.CAST, dtypes.ints+(dtypes.weakint,),src=(UPat.var("x", dtypes.bool),)), lambda x,ctx: (z3.If(ctx[1][x], 1, 0), None)),
  (UPat(Ops.CAST, dtypes.ints+(dtypes.weakint,), src=(UPat.var("x", dtypes.ints+(dtypes.weakint,)),)), lambda x,ctx: (ctx[1][x], None)),
  (UPat(Ops.CAST, dtypes.bool, name="x"), lambda x,ctx: (ctx[1][x.src[0]]!=0, None)),
  (UPat(GroupOp.ALU, name="x"), lambda x,ctx: (z3_alu[x.op](*(ctx[1][s] for s in x.src)), None)),
])

def uops_to_z3(solver:z3.Solver, *uops: UOp) -> list[z3.ExprRef]:
  lst = list(UOp.sink(*uops).toposort(gate=lambda x: x.dtype.scalar() in dtypes.ints+(dtypes.bool, dtypes.weakint) or x.op is Ops.SINK))[:-1]
  z3map: dict[UOp, z3.ExprRef] = {}
  for u in lst:
    z3_rewritten = z3_renderer.rewrite(u, ctx=(solver.ctx, z3map))
    if z3_rewritten is None: raise NotImplementedError(f"{u.op} is not supported by z3")
    new_u, constraint = cast(tuple[z3.ArithRef, z3.BoolRef|None], z3_rewritten)
    if constraint is not None: solver.add(constraint)
    z3map[u] = new_u
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
