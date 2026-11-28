from tinygrad.dtype import dtypes
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, range_str, sint
from tinygrad.helpers import strip_parens

def pretty_print(x:UOp, cache=None, d=0)->str:
  def dfs(x:UOp, cache:dict):
    for s in x.src:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(x, cache:={})
  if (cx:=cache.setdefault(x, [0,0,False]))[2]: return f"{' '*d}x{cx[0]}"
  cx[2], srcs = True, (''.join(f'\n{pretty_print(s, cache, d+2)},' for s in x.src))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{type(x).__name__}({x.op}, arg={x.argstr()}, src=({srcs}))"

# for debug
syms = { Ops.ADD: "+", Ops.SUB: "-", Ops.DIV: "//", Ops.MOD: "%", Ops.SHL: "<<", Ops.SHR: ">>",
         Ops.MUL: "*", Ops.CMPLT: "<", Ops.CMPNE: "!=", Ops.CMPEQ: "==", Ops.AND: "&", Ops.OR: "|", Ops.XOR: "^"}
# comparison operators are not in here because they are chained in python, not left-associative
precedence = {Ops.MUL:1, Ops.DIV:1, Ops.MOD:1, Ops.ADD:2, Ops.SUB:2, Ops.SHL:3, Ops.SHR:3, Ops.AND:4, Ops.XOR:5, Ops.OR:6}
def strip_binary_parens(x:UOp, left:str, right:str, code_for_op) -> str:
  if x.op not in precedence: return code_for_op(left, right)
  return code_for_op(strip_parens(left) if precedence.get(x.src[0].op,99)<=precedence[x.op] else left, strip_parens(right) if
    precedence.get(x.src[1].op,99)<precedence[x.op] else right)

# marg is ssimplify'd, so a bound can be a node this graph never contained
def marg_str(ctx, a:sint) -> str: return str(a) if not isinstance(a, UOp) else ctx[a] if a in ctx else a.render()

def render_marg(ctx,x:UOp):
  if x.op is Ops.PERMUTE: return str(x.marg)
  pieces = []
  if x.op in {Ops.RESHAPE, Ops.EXPAND}: pieces = [marg_str(ctx, a) for a in x.marg]
  if x.op in {Ops.PAD, Ops.SHRINK}: pieces = [f"({marg_str(ctx, a[0])}, {marg_str(ctx, a[1])})" for a in x.marg]
  return f"({','.join(pieces)})" if len(pieces) != 1 else f"({pieces[0]},)"

def render_index(srcs) -> str: return ''.join(f"[{strip_parens(src)}]" for src in srcs)

renderer = PatternMatcher([
  (UPat((Ops.PARAM, Ops.EXTERN), name="x"), lambda x: x.expr),
  (UPat((Ops.BUFFER, Ops.ALLOC), name="x"), lambda x:
   x.arg.name if x.arg.name is not None else f"{'a' if x.op is Ops.ALLOC else 'b'}{x.arg.slot}"),
  (UPat(Ops.AFTER, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat((Ops.SPECIAL), name="x"), lambda x: x.arg),
  (UPat(Ops.RANGE, dtypes.void, name="x"), lambda x: f"loop{x.axis_id[0]}"),
  (UPat(Ops.RANGE, name="x"), lambda x: f"r{range_str(x)}"),
  (UPat(Ops.CONST, name="x"), lambda x: str(x.val)),
  # CAST states the width, the weak CONST carries the value
  (UPat.cvar("c").cast(), lambda c: str(c.val)),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({str(x.dtype)[7:]})({ctx[x.src[0]]})"),
  (UPat(Ops.NEG, name="x"), lambda ctx,x: f"(-{ctx[x.src[0]]})"),
  (UPat(Ops.RECIP, name="x"), lambda ctx,x: f"(1/{ctx[x.src[0]]})"),
  (UPat((Ops.MAX, Ops.TRUNCDIV, Ops.TRUNCMOD), name="x"), lambda ctx,x: f"{x.op.name.lower()}({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.MULACC, name="x"), lambda ctx,x: f"({ctx[x.src[0]]}*{ctx[x.src[1]]}+{ctx[x.src[2]]})"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x: f"({ctx[x.src[1]]} if {ctx[x.src[0]]} else {ctx[x.src[2]]})"),
  (UPat(GroupOp.Movement, name="x"), lambda ctx,x: f"{ctx[x.src[0]]}.{x.op.name.lower()}({render_marg(ctx, x)})"),
  (UPat(set(syms.keys()), name="x"), lambda ctx,x: strip_binary_parens(x, ctx[x.src[0]], ctx[x.src[1]], lambda a,b: f"({a}{syms[x.op]}{b})")),
  (UPat((Ops.INDEX, Ops.STAGE), name="x"), lambda x, ctx: render_index(ctx[y] for y in x.src[1:])),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, name="idx"),)), lambda ctx,idx: f"{ctx[idx.src[0]]}{ctx[idx]}"),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, name="idx"), UPat(name="alt"), UPat(name="gate"))),
   lambda ctx,idx,alt,gate: f"({ctx[idx.src[0]]}{ctx[idx]} if {ctx[gate]} else {ctx[alt]})"),
  (UPat(Ops.STACK, name="x"), lambda ctx,x: f"{{{','.join([ctx[y] for y in x.src])}}}"),
  (UPat(GroupOp.All, name="x"), lambda x: str(x)),
])

renderer_infer = PatternMatcher([
  (UPat((Ops.MOD, Ops.DIV), name="x"), lambda ctx,x: f"{x.op.name.lower()}({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.CAST, name="x"),
    lambda ctx,x: f"{'float' if dtypes.is_float(x.dtype) else 'bool' if x.dtype is dtypes.bool else 'int'}({ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast({ctx[x.src[0]]}, {x.src[0].dtype!r}, {x.dtype!r})"),
]) + renderer

def _render_with_splits(lst:list[UOp], pm:PatternMatcher, to_render:set[UOp], split_depth:int=100) -> dict[str, str]:
  r: dict[UOp, str] = {}
  ret: dict[str, str] = {}
  depth: dict[UOp, int] = {}
  for i,u in enumerate(lst):
    # limit inline depth to avoid "too many nested parentheses" in Python parser
    op_depth = 1 + max([depth.get(s, 0) for s in u.src], default=0)
    if op_depth > split_depth: to_render.add(u)
    depth[u] = 0 if u in to_render else op_depth
    ren = pm.rewrite(u, ctx=r)
    assert isinstance(ren, str)
    if u not in to_render: r[u] = ren
    else:
      r[u] = f"c{i}" if u is not lst[-1] else "ast"
      ret[r[u]] = ren
  return ret
