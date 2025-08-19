# this converts a lowerer program into a vectorized program

import functools, itertools, operator
from tinygrad.helpers import AMX, dedup, flatten, all_same, prod
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp

def _expand_arg_to_idx(args:tuple[tuple[int, int], ...], rpk:dict[int, int]) -> int:
  idx, mul = 0, 1
  for axis,m in args[::-1]:
    idx += rpk[axis] * mul
    mul *= m
  return idx

def _choices_from_args(args:tuple[tuple[int, int], ...]) -> list[dict[int, int]]:
  return [dict(x) for x in itertools.product(*[zip(itertools.repeat(axis), range(m)) for axis,m in args])]

@functools.cache
def _swizzle_args(cargs:tuple[tuple[int, int], ...], eargs:tuple[tuple[int, int], ...], exclude_args:tuple[int, ...]) -> list[int]:
  return [_expand_arg_to_idx(eargs, {**rpk, **{x:0 for x in exclude_args}} if exclude_args else rpk) for rpk in _choices_from_args(cargs)]

def do_expand(root:UOp):
  expands = [x for x in root.src if x.op is Ops.UNROLL]
  if len(expands) == 0: return None
  # NOTE: we 0 out the reduce axis for WMMA. in theory they should all be the same, but is this always correct?
  exclude_args = tuple(dedup(root.arg[-1] + tuple(y[0] for y in flatten(root.arg[-2])))) if root.op is Ops.WMMA else ()
  if all_same(expands_args:=[x.arg for x in expands]) and len(exclude_args) == 0:
    # if there's only one expand arg, it's okay to use it (optimization)
    expand_args = expands[0].arg
  else:
    # otherwise, we sort them and GEP
    expand_args = tuple(x for x in sorted(dedup(flatten(expands_args))) if x[0] not in exclude_args)
  expand_sz = prod([x[1] for x in expand_args])
  new_srcs = []
  for i,src in enumerate(root.src):
    if src.op is Ops.UNROLL:
      if root.op is Ops.IF and i == 0:
        # IF means OR on first arg to IF
        new_srcs.append(functools.reduce(operator.__or__, [src.src[0].gep(i) for i in range(expand_sz)]))
      elif expand_args == src.arg:
        # just remove the expand
        new_srcs.append(src.src[0])
      else:
        lst = _swizzle_args(expand_args, src.arg, exclude_args)
        # if the base dtype is > 1, put those at the end
        if src.dtype.count > 1: lst = flatten([[i*src.dtype.count+j for j in range(src.dtype.count)] for i in lst])
        new_srcs.append(src.src[0].gep(tuple(lst)))
    else:
      # non-UNROLL input
      if root.op is Ops.IF:
        # for the first arg of IF, just pass them through ignoring UNROLLS
        new_srcs.append(src)
      elif root.op in {Ops.REDUCE, Ops.STORE} and src.op is Ops.RANGE:
        # for any range args of REDUCE, pass them through
        new_srcs.append(src)
      elif src.dtype.count > 1:
        # put any input dtype > 1 grouped together
        new_srcs.append(UOp(Ops.CAT, src.dtype.scalar().vec(expand_sz*src.dtype.count), (src,)*expand_sz))
      else:
        # repeat the arg
        new_srcs.append(src.broadcast(expand_sz))

  new_arg = root.arg
  if root.op is Ops.GEP:
    assert root.dtype.count == 1
    # is this right?
    new_arg = tuple(range(root.arg[0], new_srcs[0].dtype.count, new_srcs[0].dtype.count // expand_sz))
  nsrc = UOp(root.op, root.dtype.scalar().vec(root.dtype.count*expand_sz), tuple(new_srcs), new_arg)
  return UOp(Ops.UNROLL, root.dtype, (nsrc,), expand_args)

def do_contract(con:UOp):
  ex = con.src[0]
  # CONTRACT without UNROLL repeats the element VECTORIZED
  if ex.op is not Ops.UNROLL: return UOp(Ops.VECTORIZE, con.dtype, con.src*con.dtype.count)
  # CONTRACT may remove several axes from UNROLL
  assert con.dtype.count == prod([x[1] for x in con.arg]), "dtype is wrong"
  idxs = []
  for rpk in _choices_from_args(new_ex_args:=tuple(x for x in ex.arg if x not in con.arg)):
    idxs += [_expand_arg_to_idx(ex.arg, {**rpk, **lrpk}) for lrpk in _choices_from_args(con.arg)]
  return UOp(Ops.UNROLL, con.dtype, (ex.src[0].gep(tuple(idxs)),), new_ex_args)

expander = PatternMatcher([
  # double expand
  (UPat(Ops.UNROLL, name="outer", src=(UPat(Ops.UNROLL, name="inner"),)),
   lambda outer, inner: UOp(Ops.UNROLL, outer.dtype, (inner.src[0],), inner.arg+outer.arg)),
  # do expansion
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.WMMA, Ops.LOAD, Ops.STORE, Ops.INDEX,
         Ops.VECTORIZE, Ops.IF, Ops.REDUCE), name="root", custom_early_reject=set([Ops.UNROLL])), do_expand),
  (UPat(Ops.CONTRACT, name="con"), do_contract),
  # BARRIERs aren't actually expanded
  (UPat(Ops.BARRIER, src=(UPat(Ops.UNROLL, name="ex"),)),
   lambda ex: UOp(Ops.UNROLL, src=(UOp(Ops.BARRIER, src=ex.src),)*len(ex.src), arg=ex.arg)),
  # empty UNROLL is NOOP
  (UPat(Ops.UNROLL, src=(UPat.var('x'),), arg=()), lambda x: x),
  # UNROLL GEP (needed for WMMA, generalize this) -> vectorized ALU
  (UPat(Ops.UNROLL, name="ex", src=tuple(UPat.var('x').gep(i)+UPat.var('y').gep(i) for i in range(256 if AMX else 8))),
    lambda ex,x,y: UOp(Ops.UNROLL, ex.dtype, tuple((x+y).gep(i) for i in range(256 if AMX else 8)), ex.arg)),
])

def create_gate(root:UOp) -> UOp|None:
  @functools.cache
  def _gate_srcs(u:UOp, gate:UOp) -> UOp:
    if u.op is Ops.BARRIER: return u
    if u.op is Ops.LOAD and u.src[-1].op is Ops.BARRIER:
      return UOp(u.op, u.dtype, u.src[:-1]+(UOp(Ops.IF, src=(gate, u.src[-1])),), arg=u.arg)
    return u if (replace_source:=tuple(_gate_srcs(x, gate) for x in u.src)) == u.src else UOp(u.op, u.dtype, replace_source, u.arg)
  idx = root.src[0]
  if idx.op is Ops.CAST: idx = idx.src[0]
  return None if idx.op is not Ops.INDEX or len(idx.src) == 2 or (ret:=_gate_srcs(root, idx.src[2])) is root else ret

migrate_indexing = PatternMatcher([
  # create gate MUST BE BEFORE expander
  (UPat(Ops.STORE, name="root"), create_gate),
])
