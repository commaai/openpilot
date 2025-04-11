from typing import Optional, Any, Callable, cast
import functools, operator, itertools
from collections import defaultdict
from tinygrad.dtype import dtypes, ImageDType, PtrDType
from tinygrad.ops import UOp, Ops, UPat, PatternMatcher, resolve
from tinygrad.ops import graph_rewrite, GroupOp
from tinygrad.codegen.symbolic import symbolic_simple, split_uop, uop_given_valid, parse_valid, simplify_valid, sym, symbolic_flat
from tinygrad.helpers import getenv, flatten, TRANSCENDENTAL, AMX, prod, DEVECTORIZE
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, xpow, TRANSCENDENTAL_SUPPORTED_DTYPES
from tinygrad.renderer import Renderer

# ***** load/store grouping *****

def expand_index(buf:UOp, vec:UOp, mask:UOp|None=None):
  if getenv("UNSAFE_DISABLE_MASK", 0): mask = None
  # generate the individual indexes
  midx = graph_rewrite(UOp.sink(*[buf.index(vec.gep(i), mask.gep(i) if mask is not None else None) for i in range(vec.dtype.count)]),
                       symbolic_flat+load_store_indexing, name=f"index_buf_{buf.arg}")
  # extract all the relevant offsets
  offsets_rootsrc: defaultdict[Any, dict[int, list[int]]] = defaultdict(dict)
  for i in range(vec.dtype.count):
    idx: Any = midx.src[i].src[1]
    if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
    elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: root_src, arg = idx.src[1], idx.src[0].arg
    elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
    else: root_src, arg = idx, 0
    if len(midx.src[i].src) == 3: root_src = (midx.src[i].src[2], root_src)
    offsets_rootsrc[root_src].setdefault(arg, []).append(i)

  # the buf.dtype is always a pointer
  ptrdtype = cast(PtrDType, buf.dtype)

  # then rewrite everything we can into groups
  ret = []
  idxs: list[int|None] = [None]*vec.dtype.count
  global_offset = 0
  for offsets in offsets_rootsrc.values():
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for grp in grouped_offsets:
      # get the index offset for this element. using [0] is okay, because they are the same
      lidx = midx.src[offsets[grp[0]][0]]
      if len(grp) > 1: lidx = lidx.cast(ptrdtype.base.vec(len(grp)).ptr(size=ptrdtype.size, local=ptrdtype.local))
      # set the idxs of the output
      for i,g in enumerate(grp):
        for oo in offsets[g]: idxs[oo] = global_offset+i
      # add this lidx to the CAT
      ret.append(lidx)
      global_offset += len(grp)
  assert None not in idxs, f"some idxs are missing {idxs}"
  # this base thing is for image, we want the CAT to be a normal pointer
  post_cat = UOp(Ops.PTRCAT, ptrdtype.base.ptr(size=ptrdtype.size, local=ptrdtype.local).vec(vec.dtype.count), tuple(ret))
  return post_cat.gep(tuple(cast(list[int], idxs)))

def cat_after_store(cat:UOp, data:UOp):
  # TODO: this is written in many places
  offset = 0
  ret = []
  for s in cat.src:
    ret.append(s.store(data.gep(tuple(range(offset, offset+s.dtype.count)))))
    offset += s.dtype.count
  return UOp.sink(ret[0], *ret[1:])

def gep_on_store(gep:UOp, st:UOp):
  # NOTE: we need to invert the gep here, but it may be an expanding gep
  # fake argsort. TODO: handle duplicates
  a = {}
  for i,x in enumerate(gep.arg): a[x] = i
  new_arg = tuple(x[1] for x in sorted(a.items()))
  return UOp(Ops.STORE, src=(gep.src[0], st.gep(new_arg)))

load_store_folding = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat(Ops.VECTORIZE, src=UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL), name="buf")), UPat.var("vec"))), expand_index),
  (UPat(Ops.INDEX, src=(UPat(Ops.VECTORIZE, src=UPat((Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL), name="buf")), UPat.var("vec"),
                        UPat.var("mask"))), expand_index),
  # GEP after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.GEP, name="gep"),), name="ld", allow_any_len=True),
   lambda gep, ld: ld.replace(dtype=ld.dtype.scalar().vec(gep.dtype.count), src=(gep.src[0],)+ld.src[1:]).gep(gep.arg)),
  # GEP on data of STORE
  (UPat(Ops.STORE, src=(UPat(Ops.GEP, name="gep"), UPat.var("st"))), gep_on_store),
  # put PTRCAT after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.PTRCAT, name="cat"),), name="ld", allow_any_len=True),
   lambda cat,ld: UOp(Ops.CAT, ld.dtype, tuple(ld.replace(dtype=x.dtype.base, src=(x,)+ld.src[1:]) for x in cat.src))),
  # put PTRCAT after STORE
  (UPat(Ops.STORE, src=(UPat(Ops.PTRCAT, name="cat"), UPat(name="data"))), cat_after_store),
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

# *** correct load/store ***

def split_load_store(ctx:Renderer|None, ls:UOp, idx:UOp):
  if (sz:=ls.src[0].dtype.count) == 1: return None
  lengths = []
  buf = idx.src[0]
  must_divide = True
  if ctx is not None and ctx.device == "DSP":
    lengths = [128,64,32,16,8,4]
    if ls.dtype.count < 128: return None # leave these as loads (probably means something is broken)
    must_divide = False
  elif buf.dtype.base != dtypes.float and buf.dtype.base != dtypes.half and not isinstance(buf.dtype, ImageDType):
    pass
  elif isinstance(buf.dtype, ImageDType):
    lengths = [4]
  elif ctx is not None and ctx.supports_float4:
    # TODO: a better way to get this than ctx
    lengths = [8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else ([16,8,4,2] if AMX else [4,2])
  lengths.append(1)  # worst case, it's not folded
  ptrdtype = cast(PtrDType, buf.dtype)
  global_offset = 0
  ret = []
  while global_offset < sz:
    for fold_length in lengths:
      if global_offset+fold_length > sz: continue
      oidx = idx.src[1] + global_offset
      if must_divide and oidx.simplify().divides(fold_length) is None: continue
      lidx = buf.index(oidx, idx.src[2] if len(idx.src) > 2 else None)
      if fold_length > 1: lidx = lidx.cast(ptrdtype.base.vec(fold_length).ptr(size=ptrdtype.size, local=ptrdtype.local))
      if ls.op is Ops.STORE: ret.append(ls.replace(src=(lidx,ls.src[1].gep(tuple(range(global_offset, global_offset+fold_length))))+ls.src[2:]))
      else: ret.append(ls.replace(src=(lidx,)+ls.src[1:], dtype=ls.dtype.scalar().vec(fold_length)))
      global_offset += fold_length
      break
  if len(ret) == 1: return None
  return UOp(Ops.CAT, ls.dtype, tuple(ret))

def image_fixup(ls:UOp):
  # normal image load or store, with the CAST from expand_index
  if ls.src[0].op is Ops.CAST and isinstance(image_dtype:=ls.src[0].src[0].dtype, ImageDType):
    assert ls.src[0].dtype.count == 4, "image must be casted to 4"
    idx = ls.src[0].src[0]
    oidx = UOp(Ops.VECTORIZE, dtypes.int.vec(2), ((idx.src[1] // 4) % image_dtype.shape[1], (idx.src[1] // (4*image_dtype.shape[1]))))
    idx = idx.replace(src=(idx.src[0], oidx)+idx.src[2:])
    return ls.replace(src=(idx,)+ls.src[1:])

  # this is an unprocessed image without a cast, aka unfoldable image load. this doesn't work for stores
  if isinstance(image_dtype:=ls.src[0].dtype, ImageDType) and ls.src[0].src[1].dtype != dtypes.int.vec(2):
    assert ls.op is Ops.LOAD, "if an image store isn't upcasted to 4, we can't store it"
    idx = ls.src[0]
    id4 = idx.src[1] % 4
    oidx = UOp(Ops.VECTORIZE, dtypes.int.vec(2), ((idx.src[1] // 4) % image_dtype.shape[1], (idx.src[1] // (4*image_dtype.shape[1]))))
    idx = idx.replace(src=(idx.src[0], oidx)+idx.src[2:])
    vec_load = ls.replace(dtype=ls.dtype.vec(4), src=(idx,)+ls.src[1:])
    return functools.reduce(lambda ret, i: id4.ne(i).where(ret, vec_load.gep(i)), range(4), ls.const_like(float('nan')))

  return None

correct_load_store = PatternMatcher([
  # split LOAD/STORE
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.CAST, src=(UPat(Ops.INDEX, name="idx"),)),), name="ls", allow_any_len=True), split_load_store),
  # image indexing, including unfoldable images
  (UPat((Ops.LOAD, Ops.STORE), name="ls"), image_fixup),
])

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

def no_vectorized_acc(acc:UOp):
  if acc.dtype.count == 1: return None
  alus = tuple(UOp(acc.op, acc.dtype.scalar(),
    tuple(s.gep(i) if j == 0 else s for j,s in enumerate(acc.src)), acc.arg+(i,)) for i in range(acc.dtype.count))
  return UOp(Ops.VECTORIZE, acc.dtype, alus)

devectorize = PatternMatcher([
  # no ALU on vectorized dtypes
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN), name="alu"), no_vectorized_alu),
  (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
  (UPat(Ops.DEFINE_ACC, name="acc"), no_vectorized_acc),
])

def delete_redundant_gates(buf:UOp, idx:UOp, val:UOp, store_gate:UOp, cast:UOp|None=None) -> UOp|None:
  if store_gate not in [gate.src[0] for gate in val.toposort if gate.op is Ops.IF]: return None
  # remove the gate from the index
  return UOp.store(buf.index(idx).cast(cast.dtype) if cast is not None else buf.index(idx), val)

load_store_indexing = PatternMatcher([
  # simplify valid
  (UPat(Ops.AND, name="valid"), simplify_valid),
  # image load valid idx simplification
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("start_idx"), UPat.var("valid"))), simplify_valid_load),
  # index True is just Index
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("start_idx"), UPat(Ops.CONST, arg=True))), lambda buf,start_idx: buf.index(start_idx)),
  # delete_redundant_gates (after expand)
  (UPat(Ops.STORE, src=(UPat.any(stidx:=UPat.var("buf").index(UPat.var("idx"), UPat.var("store_gate")), stidx.cast().named("cast")),
                                  UPat.var("val"))), delete_redundant_gates),
])

pm_render = PatternMatcher([
  # for rendering, we use explicit VECTORIZE
  (UPat(Ops.CONST, name='c'),
   lambda c: UOp(Ops.VECTORIZE, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(Ops.VCONST, name='c'), lambda c: UOp(Ops.VECTORIZE, c.dtype, tuple(UOp.const(c.dtype.scalar(), x) for x in c.arg))),
  (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.VECTORIZE, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: gep.src[0] if gep.src[0].dtype.vcount == 1 and gep.arg == (0,) else None),
  (UPat(Ops.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
  # give any loads that are masked an alt value
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat())).or_casted(),), allow_any_len=True, name="x"),
   lambda x: x.replace(src=(x.src[0], x.const_like(0))+x.src[1:]) if len(x.src) == 1 or x.src[1].op is Ops.CUSTOM else None),
  # gate any stores that aren't gated with ifs
  (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat(src=(UPat(), UPat(), UPat(dtype=dtypes.bool)), name="idx").or_casted(), UPat()), name="store"),
    lambda store,idx: UOp(Ops.STORE, src=store.src+(UOp(Ops.IF, src=(idx.src[2],)),))),
])

# *** uop graph ***

def full_graph_rewrite(sink:UOp, opts:Optional[Renderer]=None) -> UOp:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"
  supported_ops = tuple(opts.code_for_op.keys()) if opts is not None else ()
  extra_matcher = opts.extra_matcher if opts is not None and opts.extra_matcher is not None else PatternMatcher([])

  # devectorize is optional
  if DEVECTORIZE >= 2: sink = graph_rewrite(sink, sym+load_store_folding+load_store_indexing, ctx=opts)
  elif DEVECTORIZE: sink = graph_rewrite(sink, sym+devectorize+load_store_folding+correct_load_store+load_store_indexing, ctx=opts)
  else: sink = graph_rewrite(sink, sym+load_store_folding+correct_load_store+load_store_indexing, ctx=opts)

  # optional pre matcher
  if opts is not None and opts.pre_matcher is not None: sink = graph_rewrite(sink, opts.pre_matcher)

  # final rules for the renderer (without sym)
  sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL>=2)+pm_render+extra_matcher)
  return sink
