from typing import Any, Callable
import itertools, inspect, functools, types
from tinygrad.helpers import partition, dedup, Context
from tinygrad.uop.ops import UPat, UOp, Ops, PatternMatcher, graph_rewrite, deconstruct_function

class UPatCompileError(Exception): pass

# **** UPat compiled ****
# This file builds an IR of match predicates and compiles them to Python source.
# Ops used: NATIVE (format-string predicates and inline fragments), STORE (bind a matched UOp to a name),
#           PYLITERAL (Python literal for NATIVE operands), AND/OR (clause combininers).

def _get_clause(self:UPat, base:UOp, depth=0) -> UOp:
  # build the and_clause for acceptance
  and_clause:list[UOp] = []
  if self.op is not None:
    if len(self.op) != 1:
      and_clause.append(UOp.native("{0}.op in {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=frozenset(self.op))))
    else: and_clause.append(UOp.native("{0}.op is {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=self.op[0])))
  if self.arg is not None:
    if isinstance(self.arg, int): and_clause.append(UOp.native("{0}.arg == "+str(int(self.arg)), fmt=True).call(base))
    else: and_clause.append(UOp.native("{0}.arg == {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=self.arg)))
  if self.strict_length or self.required_len > 0:
    and_clause.append(UOp.native("len({0}.src)"+(" == " if self.strict_length else " >= ")+str(self.required_len), fmt=True).call(base))
  if self.name is not None: and_clause.append(UOp(Ops.STORE, src=(UOp.native(self.name, fmt=True).call(inline=True), base)))
  if self.match_dtype is not None:
    if len(self.match_dtype) != 1:
      and_clause.append(UOp.native("{0}.dtype in {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=frozenset(self.match_dtype))))
    else:
      and_clause.append(UOp.native("{0}.dtype is {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=self.match_dtype[0])))
  if self.is_any:
    and_clause.append(UOp(Ops.OR, src=tuple(_get_clause(s, base, depth) for s in self.src[0])))
  elif self.src is not None:
    # single match
    if len(self.src) == 1 and isinstance(self.src[0], tuple):
      and_clause += [_get_clause(s, base.index(i), depth) for i,s in enumerate(self.src[0])]
    # repeat match
    elif len(self.src) == 1 and isinstance(self.src[0], itertools.repeat):
      it = UOp.native(f"ituop{depth}", fmt=True).call(inline=True)
      match = _get_clause(next(self.src[0]), it, depth+1)
      # lift named binds out of the repeat: they bind the first src, and every element must be identical to it
      stores, pred = partition(match.split_uop(Ops.AND), lambda x: x.op is Ops.STORE)
      for st in stores:
        first = st.src[1].substitute({it: base.index(0)})
        pred.append(UOp.native("{0} is {1}", fmt=True).call(st.src[1], first))
        and_clause.append(UOp(Ops.STORE, src=(st.src[0], first)))
      if len(stores): and_clause.append(UOp.native("len({0}.src) != 0", fmt=True).call(base))
      # Short-circuit before a later source can reference a capture from a mismatched first source.
      and_clause.append(UOp.native("all({0} for {1} in {2}.src)", fmt=True).call(UOp(Ops.AND, src=tuple(pred)), it, base))
    # multi match (fork)
    elif len(self.src) > 1 and all(isinstance(x, tuple) for x in self.src):
      fork_cond = [UOp(Ops.AND, src=tuple([_get_clause(s, base.index(i), depth) for i,s in enumerate(ss)])) for ss in self.src]
      and_clause.append(UOp(Ops.OR, src=tuple(fork_cond)))
    else: raise RuntimeError("broken")
  return UOp(Ops.AND, src=tuple(and_clause) or (UOp.native("True", fmt=True).call(inline=True),))

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

  # handle stores
  stores, new_src = partition(new_src, lambda x: x.op is Ops.STORE)
  if len(stores):
    if len(or_clause):
      # push stores to the top if we have an or_clause
      assert len(or_clause) == 1 and all(x.op is Ops.AND for x in or_clause[0].src)
      or_clause = [UOp(Ops.OR, src=tuple([x.replace(src=x.src+tuple(stores)) for x in or_clause[0].src]))]
      found = True
    else:
      # check for duplicate stores
      dict_stores: dict[UOp, UOp] = {}
      for store in stores:
        if store.src[0] in dict_stores:
          # duplicate store is an identity compare
          new_src.append(UOp.native("{0} is {1}", fmt=True).call(dict_stores[store.src[0]], store.src[1]))
          found = True
        else:
          dict_stores[store.src[0]] = store.src[1]
      # put the stores back
      for k,v in dict_stores.items(): new_src.append(UOp(Ops.STORE, src=(k,v)))

  # reassemble, if there's any deduping to do, do it
  if len(dretand:=dedup(new_src+or_clause)) != len(new_src)+len(or_clause): found = True
  return UOp(Ops.AND, src=tuple(dretand)) if found else None

# processor
pm_proc = PatternMatcher([(UPat(Ops.AND, name="a"), do_process_and)], compiled=False)

# renderer
def wrap(ctx, x) -> UOp:
  ctx[ret:=f"a{len(ctx)}"] = x.arg
  return UOp.native(ret, fmt=True).call(inline=True)

pm_renderer = PatternMatcher([
  (UPat(Ops.PYLITERAL, name="x"), wrap),

  # AND/OR of inline NATIVE fragments becomes a single NATIVE
  (UPat((Ops.AND, Ops.OR), src=UPat(Ops.CALL), name="x"), lambda x:
   UOp.native("(" + f" {x.op.name.lower()} ".join(y.body.arg.code for y in x.src) + ")", fmt=True).call(inline=True)
   if all(y.arg.inline for y in x.src) else None),

  (UPat(Ops.CALL, name="x"), lambda x: UOp.native(x.body.arg.code.format(*[y.body.arg.code for y in x.src[1:]]), fmt=True).call(inline=True)
   if not x.arg.inline and all(y.op is Ops.CALL and y.arg.inline for y in x.src[1:]) else None),
  (UPat(Ops.INDEX, src=(UPat(Ops.CALL, name="x"), UPat(Ops.CONST, name="c"))),
   lambda x,c: UOp.native(x.body.arg.code + f".src[{c.val}]", fmt=True).call(inline=True) if x.arg.inline else None)
], compiled=False)

def _final_render(x:UOp, has_ctx:bool, depth=1) -> list[str]:
  # if the whole clause collapsed to a single predicate (no binds), rewrap it
  if x.op is Ops.CALL and x.arg.inline: x = UOp(Ops.AND, (x,))
  assert x.op is Ops.AND
  and_pieces, store_pieces = [], []
  or_pieces: list[str] = []
  for s in x.src:
    if s.op is Ops.OR:
      assert len(or_pieces) == 0 and len(s.src) >= 1
      for ss in s.src: or_pieces.extend(_final_render(ss, has_ctx, depth+1))
    elif s.op is Ops.STORE:
      assert all(y.op is Ops.CALL and y.arg.inline for y in s.src)
      store_pieces.append(f"{s.src[0].body.arg.code}={s.src[1].body.arg.code}")
    elif s.op is Ops.CALL and s.arg.inline: and_pieces.append(s.body.arg.code)
    else: raise UPatCompileError(f"can't compile this {s}")
  # if we have an or, render it
  if len(or_pieces):
    assert len(store_pieces) == 0
    and_clause = ' and '.join(and_pieces)
    return [f"{'  '*depth}if {and_clause if len(and_clause) else 'True'}:"] + or_pieces
  # if we don't, this is a final return
  store_clause = ', '.join((["ctx=ctx"] if has_ctx else [])+store_pieces)
  and_clause = ' and '.join(and_pieces + [f"(_ret:=_fxn({store_clause})) is not None"])
  return [f"{'  '*depth}if {and_clause}: return _ret"]

def _get_code(self:UPat, has_ctx:bool):
  ret = _get_clause(self, UOp.native("uop", fmt=True).call(inline=True))
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
def upat_compile(self:UPat, fxn) -> Callable:
  real_fxn = types.FunctionType(*deconstruct_function(fxn))
  code = _get_code(self, 'ctx' in inspect.signature(real_fxn).parameters)
  if code is None: raise UPatCompileError(f"can't compile pattern defined at {self.location[0]}:{self.location[1]}")
  code_str, dyn_lookup = code
  globs = dyn_lookup.copy()
  globs["_fxn"] = real_fxn
  namespace: dict = {}
  exec(code_str, globs, namespace)  # pylint: disable=W0122
  return namespace["compiled_match"]
