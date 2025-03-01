from typing import Optional, Any, Callable
import functools, operator
from collections import defaultdict
from tinygrad.dtype import dtypes, ImageDType, PtrDType
from tinygrad.ops import UOp, Ops, UPat, PatternMatcher, resolve
from tinygrad.ops import graph_rewrite, GroupOp
from tinygrad.codegen.symbolic import symbolic_simple, split_uop, uop_given_valid, parse_valid, simplify_valid, sym
from tinygrad.helpers import getenv, flatten, dedup, TRANSCENDENTAL, AMX, prod, DEVECTORIZE
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, xpow, TRANSCENDENTAL_SUPPORTED_DTYPES
from tinygrad.renderer import Renderer

# ***** float4/image store handling *****

def fold_expanded(ex, buf):
  new_srcs = dedup(list(ex.src))
  old_new_srcs = new_srcs[:]
  is_load, is_image = new_srcs[0].op is Ops.LOAD, isinstance(buf.dtype, ImageDType)

  # TODO: get the device from the buffer somehow
  # NOTE: this can't be Device.DEFAULT because it opens devices
  if buf.dtype.base != dtypes.float and buf.dtype.base != dtypes.half and not isinstance(buf.dtype, ImageDType): return None
  lengths = [4] if is_image else ([8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else ([16,8,4,2] if AMX else [4,2]))

  # first, extract all the relevant offsets
  offsets_rootsrc: defaultdict[Any, dict] = defaultdict(dict)
  for i,s in enumerate(new_srcs):
    idx = s.src[0].src[1]
    if s.dtype.count != 1 or (is_image and idx.dtype.count == 2): continue
    if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
    elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
    else: root_src, arg = idx, 0
    # add gates for gated
    if len(s.src[0].src) == 3: root_src = (s.src[0].src[2], root_src)
    assert arg not in offsets_rootsrc[root_src], f"{offsets_rootsrc[root_src][arg]} != {i} with {len(s.src)} sources"
    offsets_rootsrc[root_src][arg] = i

  # then rewrite everything we can
  used: set[tuple[UOp, UOp]] = set()
  for rootsrc, offsets in offsets_rootsrc.items():
    for o in offsets:
      for fold_length in lengths:
        if all((rootsrc,o+i) not in used and o+i in offsets for i in range(fold_length)):
          load_1 = new_srcs[offsets[o]]
          new_src = list(load_1.src)
          oidx = new_src[0].src[1]
          if oidx.divides(fold_length) is None: continue
          if is_image:
            # for images, we rewrite the index. it must evenly divide 4 from the above check
            new_src[0] = buf.index(
              UOp(Ops.VECTORIZE, dtypes.int.vec(2), ((oidx // 4) % buf.dtype.shape[1], (oidx // (4*buf.dtype.shape[1])))),
              rootsrc[0] if isinstance(rootsrc, tuple) else None)
          else:
            # for non image, we upcast the index pointer
            new_src[0] = new_src[0].cast(new_src[0].dtype.base.vec(fold_length).ptr(size=new_src[0].dtype.size, local=new_src[0].dtype.local))
          # generate the folded new_srcs
          if is_load:
            new_load = UOp(Ops.LOAD, load_1.dtype.vec(fold_length), tuple(new_src))
            for i in range(fold_length): new_srcs[offsets[o+i]] = new_load.gep(i)
          else: # vectorize the store
            new_src[1] = UOp(Ops.VECTORIZE, new_src[1].dtype.vec(fold_length), tuple(new_srcs[offsets[o+i]].src[1] for i in range(fold_length)))
            for i in range(fold_length): new_srcs[offsets[o+i]] = UOp(Ops.STORE, dtypes.void, tuple(new_src)) if i == 0 else None
          used.update((rootsrc,o+i) for i in range(fold_length))

  # dedup expand for LOAD
  if is_load and len(old_new_srcs) != len(ex.src): new_srcs = [new_srcs[old_new_srcs.index(s)] for s in ex.src]
  # remove Nones for STORE
  return UOp(ex.op, ex.dtype, tuple(x for x in new_srcs if x is not None), ex.arg) if len(used) else None

def fix_unfoldable_image_load(load:UOp, buf:UOp):
  if not isinstance(buf.dtype, ImageDType) or (oidx:=load.src[0].src[1]).dtype.count == 2: return None
  id4 = oidx % 4
  new_src = list(load.src)
  # TODO: copied logic from above
  new_src[0] = load.src[0].src[0].index(
    UOp(Ops.VECTORIZE, dtypes.int.vec(2), ((oidx // 4) % buf.dtype.shape[1], (oidx // (4*buf.dtype.shape[1])))),
    load.src[0].src[2] if len(load.src[0].src) == 3 else None)
  vec_load = UOp(Ops.LOAD, load.dtype.vec(4), tuple(new_src))
  return functools.reduce(lambda ret, i: id4.ne(i).where(ret, vec_load.gep(i)), range(4), load.const_like(float('nan')))

buf_idx_pat = UPat(Ops.INDEX, src=(UPat.var("buf"),), allow_any_len=True)
float4_folding = PatternMatcher([
  (UPat(Ops.VECTORIZE, src=UPat(Ops.LOAD, src=(buf_idx_pat,), allow_any_len=True), name="ex"), fold_expanded),
  (UPat((Ops.BARRIER, Ops.SINK), src=UPat(Ops.STORE, src=(buf_idx_pat,), allow_any_len=True), name="ex"), fold_expanded),
])

# ***** image load valid simplification *****

def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> UOp|None:
  if (idx:=uop_given_valid(valid, start_idx)) is None: return buf.const_like(0)
  if not isinstance(buf.dtype, ImageDType): return None if idx is start_idx else buf.index(idx, valid)

  # wait for it to be image indexed before running simplification
  if start_idx.dtype.count != 2: return None

  # can drop valid if idx is out of bound when valid is False
  drop_stmt = []
  for stmt in split_uop(valid, Ops.AND):
    X, is_upper_bound, c = parse_valid(stmt)

    # for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
    if not is_upper_bound and c == 1 and all(u.op in GroupOp.Irreducible and u.vmin == 0 for u in split_uop(X, Ops.ADD)):
      testidx = functools.reduce(lambda nowidx,u: nowidx.substitute({u:u.const_like(0)}), split_uop(X, Ops.ADD), idx)
      testidx = testidx.simplify()
      if testidx.gep(0).vmax < 0 or testidx.gep(1).vmax < 0:
        drop_stmt.append(stmt)
        continue

    # if X <= c, check if it's out of bound when X = c+1
    # if X >= c, check if it's out of bound when X = c-1
    test_value = c + 1 if is_upper_bound else c - 1
    for i,b in zip(idx.src, (buf.dtype.shape[1], buf.dtype.shape[0])):
      if i.is_increasing():
        rw = i.substitute({X:X.const_like(test_value)}).simplify()
        if rw.vmin >= b or rw.vmax < 0:
          drop_stmt.append(stmt)
          break

  if not drop_stmt and idx is start_idx: return None
  new_valid = functools.reduce(operator.and_, ss) if (ss:=[s for s in split_uop(valid, Ops.AND) if s not in drop_stmt]) else None
  return buf.index(idx, new_valid)

# ***** optional patterns *****

powers_of_two = {2**i:i for i in range(64)}
@functools.lru_cache(None)
def get_late_rewrite_patterns(ops, force_transcendental=False):
  pat: list[tuple[UPat, Callable]] = [(UPat(op, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat.var("d"),)), f) for op,f in \
           ((Ops.EXP2, xexp2), (Ops.LOG2, xlog2), (Ops.SIN, xsin)) if op not in ops or force_transcendental]
  # rewrite SQRT to xpow 0.5
  if Ops.SQRT not in ops: pat.append((UPat(Ops.SQRT, src=UPat.var("d")), lambda d: xpow(d, d.const_like(0.5))))
  # rewrite MOD to AND (which should always be supported, but not for generic in tests): x % (2**y) -> x & (2**y-1)
  if Ops.AND in ops: pat += [(UPat.var("x", dtypes.ints)%UPat.cvar("c"), lambda x,c: x & (c.arg-1) if c.arg in powers_of_two else None)]
  # rewrite MUL/IDIV to SHL+SHR: x*(2**y) -> shl(x,y) and x//(2**y) -> shr(x,y)
  if Ops.SHL in ops: pat += [(UPat.var("x", dtypes.ints)*UPat.cvar("c"), lambda c,x: x << v if (v:=powers_of_two.get(c.arg, 0)) else None)]
  if Ops.SHR in ops:
    pat += [(UPat.var("x", dtypes.ints)//UPat.cvar("c"), lambda x,c: x >> v if (v:=powers_of_two.get(c.arg, 0)) and resolve(x>=0,False) else None)]
  if Ops.NEG in ops:
    pat += [(UPat.var('x')*-1, lambda x: x.alu(Ops.NEG))]
    if Ops.SUB in ops: pat += [(UPat.var('x')+UPat.var('y').alu(Ops.NEG), lambda x,y: x.alu(Ops.SUB, y))]
  if Ops.MULACC in ops: pat += [(UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c))]
  return PatternMatcher(pat)


# *** uop expander ***

# TODO: there's a lot shared with gep_through_wmma here
def no_vectorized_wmma(wmma:UOp):
  out_sz = prod(x[1] for x in wmma.arg[6][-1])
  if wmma.dtype.count == out_sz: return None
  tsrcs = []
  for s,sz in zip(wmma.src, wmma.arg[6]):
    ssz = prod(x[1] for x in sz)
    tsrcs.append([s.gep(tuple(range(grp, grp+ssz))) for grp in range(0, s.dtype.count, ssz)])
  wmmas = [UOp(Ops.WMMA, wmma.dtype.scalar().vec(out_sz), tsrc, wmma.arg) for tsrc in zip(*tsrcs)]
  wmma_ex = flatten([[e.gep(i) for i in range(out_sz)] for e in wmmas])
  return UOp(Ops.VECTORIZE, wmma.dtype, tuple(wmma_ex))

def no_vectorized_alu(alu):
  if alu.dtype.vcount == 1: return None
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.VECTORIZE, alu.dtype, alus)

def no_vectorized_load_store(ls:UOp):
  idx = ls.src[0]
  assert isinstance(idx.dtype, PtrDType)
  if idx.dtype.v == 1: return None
  tv = [UOp(ls.op, ls.dtype.scalar(), tuple(j.gep(i) for j in ls.src)) for i in range(idx.dtype.v)]
  return UOp(Ops.VECTORIZE, ls.dtype, tuple(tv))

def no_vectorized_acc(acc:UOp):
  if acc.dtype.count == 1: return None
  alus = tuple(UOp(acc.op, acc.dtype.scalar(),
    tuple(s.gep(i) if j == 0 else s for j,s in enumerate(acc.src)), acc.arg+(i,)) for i in range(acc.dtype.count))
  return UOp(Ops.VECTORIZE, acc.dtype, alus)

devectorize = PatternMatcher([
  # no ALU on vectorized dtypes
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.INDEX), name="alu"), no_vectorized_alu),
  (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
  (UPat(Ops.DEFINE_ACC, name="acc"), no_vectorized_acc),
  (UPat((Ops.LOAD, Ops.STORE), name="ls"), no_vectorized_load_store),
])

devectorize_load_store = PatternMatcher([
  # TODO: add vectorized support to transcendental
  (UPat((Ops.INDEX, Ops.EXP2, Ops.LOG2, Ops.SIN), name="alu"), no_vectorized_alu),
  (UPat((Ops.LOAD, Ops.STORE), name="ls"), no_vectorized_load_store),
])

def delete_redundant_gates(buf:UOp, idx:UOp, val:UOp, store_gate:UOp, cast:UOp|None=None) -> UOp|None:
  if store_gate not in [gate.src[0] for gate in val.toposort if gate.op is Ops.IF]: return None
  # remove the gate from the index
  return UOp.store(buf.index(idx).cast(cast.dtype) if cast is not None else buf.index(idx), val)

load_store_indexing = PatternMatcher([
  # late fixup of unfoldable image loads
  (UPat(Ops.LOAD, src=(UPat.var("buf"), UPat()), allow_any_len=True, name="load"), fix_unfoldable_image_load),
  # simplify valid
  (UPat(Ops.AND, name="valid"), simplify_valid),
  # image load valid idx simplification
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("start_idx"), UPat.var("valid"))), simplify_valid_load),
  # delete_redundant_gates (after expand)
  (UPat(Ops.STORE, src=(UPat.any(stidx:=UPat.var("buf").index(UPat.var("idx"), UPat.var("store_gate")), stidx.cast().named("cast")),
                                  UPat.var("val"))), delete_redundant_gates),
])

def move_mask(x:UOp, buf:UOp, idx:UOp, mask:UOp, cast:UOp|None=None) -> UOp:
  # this moves the mask from the indexing to the load/store op for rendering
  nidx = buf.index(idx).cast(cast.dtype) if cast is not None else buf.index(idx)
  return UOp.load(nidx, x.const_like(0), mask, *x.src[1:], dtype=x.dtype) if x.op is Ops.LOAD else UOp.store(nidx, x.src[1], mask, *x.src[2:])

pm_render = PatternMatcher([
  # for rendering, we use explicit VECTORIZE
  (UPat(Ops.CONST, name='c'),
   lambda c: UOp(Ops.VECTORIZE, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(Ops.VCONST, name='c'), lambda c: UOp(Ops.VECTORIZE, c.dtype, tuple(UOp.const(c.dtype.scalar(), x) for x in c.arg))),
  (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.VECTORIZE, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  (UPat(Ops.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
  # move masks of loads/stores
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat.any(masked_index:=UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx"), UPat.var("mask"))),
                                               masked_index.cast(None).named("cast")),), allow_any_len=True, name="x"), move_mask),
  # gate any stores that aren't gated with ifs
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat(), UPat(), UPat(dtype=dtypes.bool)), name="store"),
    lambda store: UOp(Ops.STORE, src=store.src[:2]+(UOp(Ops.IF, src=(store.src[2],)),))),
])

# *** uop graph ***

def full_graph_rewrite(sink:UOp, opts:Optional[Renderer]=None) -> UOp:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"
  supported_ops = tuple(opts.code_for_op.keys()) if opts is not None else ()
  extra_matcher = opts.extra_matcher if opts is not None and opts.extra_matcher is not None else PatternMatcher([])

  if DEVECTORIZE:
    # devectorize + load_store_indexing
    sink = graph_rewrite(sink, sym+(devectorize+float4_folding if opts is not None and opts.supports_float4 else devectorize)+load_store_indexing)
  else:
    # new devectorize only for load/store
    sink = graph_rewrite(sink, sym+devectorize_load_store)

  # optional pre matcher
  if opts is not None and opts.pre_matcher is not None: sink = graph_rewrite(sink, opts.pre_matcher)

  # final rules for the renderer (without sym)
  sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL>=2)+pm_render+extra_matcher)
  return sink
