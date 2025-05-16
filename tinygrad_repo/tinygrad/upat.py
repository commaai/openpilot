from typing import Any, Callable
import itertools, inspect, functools, types
from tinygrad.helpers import partition, dedup, Context
from tinygrad.ops import UPat, UPatAny, UOp, Ops, PatternMatcher, graph_rewrite, deconstruct_function

class UPatCompileError(Exception): pass

# **** UPat compiled ****

def _get_clause(self:UPat, base:UOp, depth=0) -> UOp:
  if isinstance(self, UPatAny):
    assert len(self.src) == 1
    return UOp(Ops.AND, src=(UOp(Ops.OR, src=tuple(_get_clause(s, base, depth) for s in self.src[0])),))
  # build the and_clause for acceptance
  and_clause:list[UOp] = []
  if self.op is not None:
    if len(self.op) > 1: and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=tuple(int(x) for x in self.op))), arg="{0}.op in {1}"))
    else: and_clause.append(UOp(Ops.CUSTOM, src=(base,), arg="{0}.op == "+str(self.op[0].value)))
  if self.arg is not None:
    if isinstance(self.arg, int): and_clause.append(UOp(Ops.CUSTOM, src=(base,), arg="{0}.arg == "+str(int(self.arg))))
    else: and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=self.arg)), arg="{0}.arg == {1}"))
  if self.strict_length or self.required_len > 0:
    and_clause.append(UOp(Ops.CUSTOM, src=(base,), arg=("len({0}.src)"+(" == " if self.strict_length else " >= ")+str(self.required_len))))
  if self.name is not None: and_clause.append(UOp(Ops.ASSIGN, src=(UOp(Ops.DEFINE_VAR, arg=self.name), base)))
  if self.dtype is not None:
    if len(self.dtype) > 1:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=tuple(self.dtype))), arg="({0}.dtype in {1} or {0}.dtype._scalar in {1})"))
    else: and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=self.dtype[0])), arg="({0}.dtype == {1} or {0}.dtype._scalar == {1})"))
  if self.src is not None:
    # single match
    if len(self.src) == 1 and isinstance(self.src[0], tuple):
      and_clause += [_get_clause(s, base.gep(i), depth) for i,s in enumerate(self.src[0])]
    # repeat match
    elif len(self.src) == 1 and isinstance(self.src[0], itertools.repeat):
      it = UOp(Ops.NOOP, arg=f"ituop{depth}")
      match = _get_clause(next(self.src[0]), it, depth+1)
      and_clause.append(UOp(Ops.RANGE, src=(match, it, base), arg="all([{0} for {1} in {2}.src])"))
    # multi match (fork)
    elif len(self.src) > 1 and all(isinstance(x, tuple) for x in self.src):
      fork_cond = [UOp(Ops.AND, src=tuple([_get_clause(s, base.gep(i), depth) for i,s in enumerate(ss)])) for ss in self.src]
      and_clause.append(UOp(Ops.OR, src=tuple(fork_cond)))
    else: raise RuntimeError("broken")
  return UOp(Ops.AND, src=tuple(and_clause))

# *** pattern matcher ***

def do_process_and(a:UOp) -> UOp|None:
  found = False
  new_src:list[UOp] = []
  or_clause:list[UOp] = []

  # remove any nested ANDs, extract or clauses
  for x in a.src:
    if x.op is Ops.AND:
      new_src.extend(x.src)
      found = True
    elif x.op is Ops.OR: or_clause.append(x)
    else: new_src.append(x)

  # too big to compile
  if len(or_clause) >= 4: raise UPatCompileError("too big to compile")

  # one or clause max
  if len(or_clause) > 1:
    # need the product of the or clauses
    or_clause = [UOp(Ops.OR, src=tuple([UOp(Ops.AND, src=x) for x in itertools.product(*[x.src for x in or_clause])]))]
    found = True

  # handle assigns
  assigns, new_src = partition(new_src, lambda x: x.op is Ops.ASSIGN)
  if len(assigns):
    if len(or_clause):
      # push assigns to the top if we have an or_clause
      assert len(or_clause) == 1 and all(x.op is Ops.AND for x in or_clause[0].src)
      or_clause = [UOp(Ops.OR, src=tuple([x.replace(src=x.src+tuple(assigns)) for x in or_clause[0].src]))]
      found = True
    else:
      # check for duplicate assigns
      dict_assigns: dict[UOp, UOp] = {}
      for a in assigns:
        if a.src[0] in dict_assigns:
          # duplicate assign is a compare
          new_src.append(UOp(Ops.CMPNE, src=(dict_assigns[a.src[0]], a.src[1])))
          found = True
        else:
          dict_assigns[a.src[0]] = a.src[1]
      # put the assigns back
      for k,v in dict_assigns.items(): new_src.append(UOp(Ops.ASSIGN, src=(k,v)))

  # reassemble, if there's any deduping to do, do it
  if len(dretand:=dedup(new_src+or_clause)) != len(new_src)+len(or_clause): found = True
  return UOp(Ops.AND, src=tuple(dretand)) if found else None

# processor
pm_proc = PatternMatcher([(UPat(Ops.AND, name="a"), do_process_and)], compiled=False)

# renderer
def wrap(ctx, x) -> UOp:
  ctx[ret:=f"a{len(ctx)}"] = x.arg
  return UOp(Ops.NOOP, arg=ret)

pm_renderer = PatternMatcher([
  (UPat(Ops.BIND, name="x"), wrap),

  # CMPNE is actually equal
  (UPat(Ops.CMPNE, name="x"), lambda x: UOp(Ops.CUSTOM, src=x.src, arg="{0} is {1}")),

  # RANGE can't have OR inside it
  (UPat(Ops.RANGE, src=(UPat(Ops.AND, src=UPat(Ops.NOOP), name="x"), UPat(), UPat()), name="r"),
    lambda r,x: r.replace(op=Ops.CUSTOM, src=(UOp(Ops.NOOP, arg="(" + ' and '.join(y.arg for y in x.src) + ")"),)+r.src[1:])),

  (UPat(Ops.CUSTOM, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg.format(*[y.arg for y in x.src]))),
  (UPat(Ops.GEP, src=UPat(Ops.NOOP, name="x"), name="g"), lambda x,g: x.replace(arg=x.arg+f".src[{g.arg[0]}]"))
], compiled=False)

def _final_render(x:UOp, has_ctx:bool, depth=1) -> list[str]:
  assert x.op is Ops.AND
  and_pieces, assign_pieces = [], []
  or_pieces: list[str] = []
  for s in x.src:
    if s.op is Ops.OR:
      assert len(or_pieces) == 0 and len(s.src) >= 1
      for ss in s.src: or_pieces.extend(_final_render(ss, has_ctx, depth+1))
    elif s.op is Ops.ASSIGN:
      assert s.src[0].op is Ops.DEFINE_VAR and s.src[1].op is Ops.NOOP
      assign_pieces.append(f"{s.src[0].arg}={s.src[1].arg}")
    elif s.op is Ops.NOOP: and_pieces.append(s.arg)
    else: raise UPatCompileError(f"can't compile this {s}")
  # if we have an or, render it
  if len(or_pieces):
    assert len(assign_pieces) == 0
    and_clause = ' and '.join(and_pieces)
    return [f"{'  '*depth}if {and_clause if len(and_clause) else 'True'}:"] + or_pieces
  # if we don't, this is a final return
  assign_clause = ', '.join((["ctx=ctx"] if has_ctx else [])+assign_pieces)
  and_clause = ' and '.join(and_pieces + [f"(_ret:=_fxn({assign_clause})) is not None"])
  return [f"{'  '*depth}if {and_clause}: return _ret"]

def _get_code(self:UPat, has_ctx:bool):
  ret = _get_clause(self, UOp(Ops.NOOP, arg="uop"))
  try:
    # TODO: this should be tracked in a "system" rewrite, not untracked or tracked with kernel
    with Context(TRACK_MATCH_STATS=0):
      ret = graph_rewrite(ret, pm_proc, name="process UPat")
      dyn_lookup: dict[str, Any] = {}
      out = graph_rewrite(ret, pm_renderer, ctx=dyn_lookup, name="compile UPat")
      rendered = _final_render(out, has_ctx)
  except UPatCompileError:
    #print("FAILED", self, self.location)
    return None
  return '\n'.join([f"# match for {self.location}", "def compiled_match(uop, ctx):"] + rendered + ["  return None"]), dyn_lookup

@functools.cache
def upat_compile(self:UPat, fxn) -> Callable|None:
  real_fxn = types.FunctionType(*deconstruct_function(fxn))
  code = _get_code(self, 'ctx' in inspect.signature(real_fxn).parameters)
  if code is None: return None
  code_str, dyn_lookup = code
  globs = dyn_lookup.copy()
  globs["_fxn"] = real_fxn
  namespace: dict = {}
  exec(code_str, globs, namespace)  # pylint: disable=W0122
  return namespace["compiled_match"]
