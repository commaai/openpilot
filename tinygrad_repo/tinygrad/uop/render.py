from typing import cast
from tinygrad.dtype import dtypes
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, multirange_str, range_str, consumer_map_from_toposort
from tinygrad.helpers import strip_parens, all_same

def pretty_print(x:UOp, cache=None, d=0)->str:
  def dfs(x:UOp, cache:dict):
    for s in x.src:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(x, cache:={})
  if (cx:=cache.setdefault(x, [0,0,False]))[2]: return f"{' '*d}x{cx[0]}"
  cx[2], srcs = True, (''.join(f'\n{pretty_print(s, cache, d+2)},' for s in x.src))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{type(x).__name__}({x.op}, {x.dtype}, arg={x.argstr()}{x.tagstr()}, src=({srcs}))"

# ***** uop helpers *****

def print_uops(uops:list[UOp]):
  uops_index = {u:i for i,u in enumerate(uops)}
  for i,u in enumerate(uops):
    formatted_srcs = [(uops_index[x] if x.op is not Ops.CONST else f"{x.arg}") if x in uops else "--" for x in u.src]
    print(f"{i:4d} {str(u.op):20s}: {multirange_str(u.ranges, color=True, pad=10)} {str(u.dtype):40s} " f"{str(formatted_srcs):32s} {u.arg}")

# for debug
syms = { Ops.ADD: "+", Ops.SUB: "-", Ops.FLOORDIV: "//", Ops.FLOORMOD: "%", Ops.SHL: "<<", Ops.SHR: ">>",
         Ops.MUL: "*", Ops.CMPLT: "<", Ops.CMPNE: "!=", Ops.AND: "&", Ops.OR: "|", Ops.XOR: "^"}
# comparison operators are not in here because they are chained in python, not left-associative
precedence = {Ops.MUL:1, Ops.FLOORDIV:1, Ops.FLOORMOD:1, Ops.ADD:2, Ops.SUB:2, Ops.SHL:3, Ops.SHR:3, Ops.AND:4, Ops.XOR:5, Ops.OR:6}
def strip_binary_parens(x:UOp, left:str, right:str, code_for_op) -> str:
  if x.op not in precedence: return code_for_op(left, right)
  return code_for_op(strip_parens(left) if precedence.get(x.src[0].op,99)<=precedence[x.op] else left, strip_parens(right) if
    precedence.get(x.src[1].op,99)<precedence[x.op] else right)

renderer = PatternMatcher([
  (UPat((Ops.DEFINE_VAR,), name="x"), lambda x: x.expr),
  (UPat(Ops.PARAM, src=(UPat(), UPat(), UPat(), UPat(), UPat(Ops.NOOP, name="x"))), lambda x: x.arg),
  (UPat((Ops.SPECIAL), name="x"), lambda x: x.arg),
  (UPat(Ops.RANGE, name="x"), lambda x: f"r{range_str(x)}"),
  (UPat(Ops.CONST, name="x"), lambda x: str(x.arg)),
  (UPat(Ops.UNROLL, name="x"), lambda ctx,x,u: f"UNROLL({ctx[x.src[0]]}, {u.arg})"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({str(x.dtype)[7:]})({ctx[x.src[0]]})"),
  (UPat(Ops.BIND, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat(Ops.NEG, name="x"), lambda ctx,x: f"(-{ctx[x.src[0]]})"),
  (UPat(Ops.RECIPROCAL, name="x"), lambda ctx,x: f"(1/{ctx[x.src[0]]})"),
  (UPat(Ops.MAX, name="x"), lambda ctx,x: f"max({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.MULACC, name="x"), lambda ctx,x: f"({ctx[x.src[0]]}*{ctx[x.src[1]]}+{ctx[x.src[2]]})"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x: f"({ctx[x.src[1]]} if {ctx[x.src[0]]} else {ctx[x.src[2]]})"),
  (UPat(Ops.CDIV, name="x"), lambda ctx,x: f"cdiv({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.CMOD, name="x"), lambda ctx,x: f"cmod({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(set(syms.keys()), name="x"), lambda ctx,x: strip_binary_parens(x, ctx[x.src[0]], ctx[x.src[1]], lambda a,b: f"({a}{syms[x.op]}{b})")),
  (UPat((Ops.INDEX, Ops.STAGE), name="x"), lambda x, ctx: ''.join([f"[{strip_parens(ctx[y])}]" for y in x.src[1:]])),
  (UPat(Ops.STACK, name="x"),
   lambda ctx,x: f"{{{','.join([ctx[y] for y in x.src])}}}" if not x.src or not all_same(x.src) else f"{{{ctx[x.src[0]]}, ...}}"),
  (UPat(GroupOp.All, name="x"), lambda x: str(x)),
])

renderer_infer = PatternMatcher([
  (UPat(Ops.CMOD, name="x"), lambda ctx,x: f"cmod({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.CDIV, name="x"), lambda ctx,x: f"cdiv({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.FLOORMOD, name="x"), lambda ctx,x: f"floormod({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.FLOORDIV, name="x"), lambda ctx,x: f"floordiv({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast({ctx[x.src[0]]}, {x.src[0].dtype!r}, {x.dtype!r})"),
]) + renderer

# *** pyrender ***

def srcs(ctx, src): return f"({ctx[src[0]]},)" if len(src) == 1 else f"({', '.join([ctx[x] for x in src])})"
def render_marg(ctx,x:UOp):
  if x.op is Ops.PERMUTE: return str(x.marg)
  if x.op is Ops.FLIP: return str(tuple([i for i,x in enumerate(x.marg) if x]))
  pieces = []
  if x.op in {Ops.RESHAPE, Ops.EXPAND}:
    pieces = [f"{ctx[a] if isinstance(a, UOp) else str(a)}" for a in x.marg]
  if x.op in {Ops.PAD, Ops.SHRINK}:
    pieces = [f"({ctx[a[0]] if isinstance(a[0], UOp) else str(a[0])}, {ctx[a[1]] if isinstance(a[1], UOp) else str(a[1])})" for a in x.marg]
  return f"({','.join(pieces)})" if len(pieces) != 1 else f"({pieces[0]},)"

sugar = {Ops.SINK, Ops.END, Ops.STORE, Ops.LOAD, Ops.UNIQUE, Ops.SQRT, Ops.INDEX, Ops.REDUCE, Ops.AFTER, Ops.THREEFRY,
         Ops.WHERE, Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.CONTIGUOUS, Ops.BARRIER, Ops.DETACH}
pm_pyrender_extra = PatternMatcher([
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE, name="u"), UPat(Ops.DEVICE, name="d")), name="x"),
   lambda x,u,d: f"UOp.unique_const({x.arg}, dtype={x.dtype}, device={repr(d.arg)}, unique={u.arg})"),
  (UPat(Ops.CONST, src=(UPat(Ops.DEVICE, name="d"),), name="x"), lambda x,d: f"UOp.const({x.dtype}, {x.arg}, device={repr(d.arg)})"),
  (UPat(Ops.CONST, src=(), name="x"), lambda x: f"UOp.const({x.dtype}, {x.arg})"),
  (UPat(Ops.DEFINE_VAR, src=(), name="x"), lambda x:
    f"UOp.variable(\"{x.arg[0]}\", {x.arg[1]}, {x.arg[2]}{', dtype='+str(x.dtype) if x.dtype is not dtypes.weakint else ''})"),
  (UPat((Ops.CAST, Ops.BITCAST), name="x"), lambda ctx,x: f"{ctx[x.src[0]]}.{x.op.name.lower()}({x.dtype})"),
  (UPat(Ops.SPECIAL, src=(UPat(Ops.CONST),), name="x"), lambda x: f"UOp.special({x.src[0].arg}, {repr(x.arg)}, dtype={x.dtype})"),
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE, name="u"), UPat(Ops.DEVICE, name="d")), name="x"), lambda x,u,d:
    f"UOp.new_buffer({repr(d.arg)}, {x.arg}, {x.dtype}, {u.arg})"),
  (UPat(Ops.COPY, src=(UPat(name="x"), UPat(Ops.DEVICE, name="d"))), lambda ctx,x,d: f"{ctx[x]}.copy_to_device({repr(d.arg)})"),
  (UPat(Ops.CUSTOM_FUNCTION, name="x"), lambda ctx,x: f"UOp(Ops.CUSTOM_FUNCTION, {x.dtype}, src={srcs(ctx, x.src)}, arg={x.arg!r})"),
  (UPat(Ops.REDUCE, name="r"), lambda ctx,r: f"{ctx[r.src[0]]}._rop({r.arg[0]}, {r.arg[1]})" if len(r.arg[1]) else None),
  # NOTE: range has srcs sometimes after control flow
  (UPat(Ops.RANGE, src=(UPat(Ops.CONST, name="c"),), allow_any_len=True, name="x"), lambda ctx,x,c:
    "UOp.range("+', '.join([str(c.arg)] + [repr(y) for y in x.arg])+
      (f', src={srcs(ctx, x.src[1:])}' if len(x.src) > 1 else '')+(', dtype='+str(x.dtype) if x.dtype is not dtypes.weakint else '')+")"),
  # TODO: index shouldn't mismatch dtype
  (UPat(Ops.INDEX, src=(UPat(), UPat()), allow_any_len=True, name="x"), lambda ctx,x:
   f"{ctx[x.src[0]]}.index({ctx[x.src[1]]}, "+''.join([f"{ctx[xx]}, " for xx in x.src[2:]])+
    (f"dtype={x.dtype})" if x.src[0].dtype != x.dtype else "ptr=True)") if x.src[0].dtype.base != x.dtype else None),
  # TODO: movement ops simplify stuff, this can break SPEC=2
  #(UPat(GroupOp.Movement, name="x"), lambda ctx,x: f"{ctx[x.src[0]]}.{x.op.name.lower()}({render_marg(ctx,x)})"),
  # NOTE: CMPNE doesn't work cause there's no __rne__
  # explicit trunc ops: `//` and `%` parse as FLOORDIV/FLOORMOD, so render CDIV/CMOD via .alu()
  (UPat(Ops.CDIV, name="x"), lambda ctx,x: f"{ctx[x.src[0]]}.alu(Ops.CDIV, {ctx[x.src[1]]})"),
  (UPat(Ops.CMOD, name="x"), lambda ctx,x: f"{ctx[x.src[0]]}.alu(Ops.CMOD, {ctx[x.src[1]]})"),
  # NOTE: only match CONSTs without UNIQUE (len(src)==1), unique_const needs explicit rendering
  (UPat(set(syms.keys())-{Ops.SUB, Ops.CMPNE, Ops.CDIV, Ops.CMOD}, src=(UPat(Ops.CONST, src=(UPat(Ops.DEVICE),), name="y"), UPat(name="z")),
        name="x"), lambda ctx,x,y,z: strip_binary_parens(x, str(y.arg), ctx[z], lambda a,b: f"({a}{syms[x.op]}{b})") if y.device==z.device else None),
  # NOTE: sub doesn't work cause it's written as add/mul
  (UPat(set(syms.keys())-{Ops.SUB, Ops.CDIV, Ops.CMOD}, src=(UPat(name="y"), UPat(Ops.CONST, src=(UPat(Ops.DEVICE),), name="z")), name="x"),
    lambda ctx,x,y,z: strip_binary_parens(x, ctx[y], str(z.arg), lambda a,b: f"({a}{syms[x.op]}{b})") if y.device==z.device else None),
  (UPat(set(syms.keys())-{Ops.SUB, Ops.CDIV, Ops.CMOD}, name="x"), lambda ctx,x:
    strip_binary_parens(x, ctx[x.src[0]], ctx[x.src[1]], lambda a,b: f"({a}{syms[x.op]}{b})")),
  (UPat(sugar, src=(), name="x"), lambda x: f"UOp.{x.op.name.lower()}("+', '.join(([f'arg={repr(x.arg)}'] if x.arg is not None else []))+")"),
  (UPat(sugar, name="x"), lambda ctx,x: f"{ctx[x.src[0]]}.{x.op.name.lower()}("+', '.join([ctx[y] for y in x.src[1:]] + \
    ([f'arg={repr(x.arg)}'] if x.arg is not None else []))+")"),
])

# NOTE: you can remove pm_pyrender_extra and it'll still be correct
pm_pyrender = pm_pyrender_extra+PatternMatcher([
  (UPat(GroupOp.All, name="u"), lambda ctx,u: f"UOp({u.op}, {u.dtype}, {srcs(ctx,u.src)}"+(f", {repr(u.arg)})" if u.arg is not None else ")")),
])

def _render_with_splits(lst:list[UOp], pm:PatternMatcher, to_render:set[UOp], split_depth:int=100) -> dict[str, str]:
  r: dict[UOp, str] = {}
  ret: dict[str, str] = {}
  depth: dict[UOp, int] = {}
  for i,u in enumerate(lst):
    # limit inline depth to avoid "too many nested parentheses" in Python parser
    op_depth = 1 + max([depth.get(s, 0) for s in u.src], default=0)
    if op_depth > split_depth: to_render.add(u)
    depth[u] = 0 if u in to_render else op_depth
    ren = cast(str, pm.rewrite(u, ctx=r))
    assert isinstance(ren, str)
    if u.tag is not None: ren += f".rtag({repr(u.tag)})"
    if u not in to_render: r[u] = ren
    else:
      r[u] = f"c{i}" if u is not lst[-1] else "ast"
      ret[r[u]] = ren
  return ret

def pyrender(ast:UOp) -> str:
  lst = list(ast.toposort())

  cmap = consumer_map_from_toposort(lst)
  not_rendered = {Ops.CONST, Ops.DEVICE}
  always_rendered = {Ops.PARAM, Ops.LOAD, Ops.SPECIAL, Ops.RANGE, Ops.CONTIGUOUS, Ops.STACK,
                     Ops.BUFFER, Ops.COPY, Ops.CALL, Ops.FUNCTION, Ops.WHERE, Ops.END}

  to_render: set[UOp] = {ast}
  for u in lst:
    if u.op in {Ops.SINK}:
      for s in u.src: to_render.add(s)
    if u.op is Ops.STORE: to_render.add(u.src[1])
    if u.op is Ops.REDUCE: to_render.add(u.src[0])
    if u.op in {Ops.CALL, Ops.FUNCTION}: raise NotImplementedError("call can't be pyrendered")
    if u.op in not_rendered: continue
    # checking the consumers is not enough, you have to make sure it's not used twice by the one consumer
    if len(cmap[u]) == 1 and len([x for x in list(cmap[u].keys())[0].src if x is u]) == 1 and u.op not in always_rendered: continue
    to_render.add(u)

  ret = _render_with_splits(lst, pm_pyrender, to_render)
  return '\n'.join([f"{k} = {strip_parens(v)}" for k,v in ret.items()])
