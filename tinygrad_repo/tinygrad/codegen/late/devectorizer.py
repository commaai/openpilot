from typing import Any, cast
import functools, itertools
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.dtype import dtypes, ImageDType, DType, AddrSpace, Invalid, PtrDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp, identity_element
from tinygrad.uop.symbolic import uop_given_valid, parse_valid, invalid_gate
from tinygrad.helpers import getenv, flatten, prod
from tinygrad.renderer import Renderer

# ***** image load valid simplification *****

@functools.cache
def _drop_valid_stmts(valid:UOp, idx:UOp, height:int, width:int) -> list[UOp]:
  # can drop valid if idx is out of bound when valid is False
  drop_stmt = []
  for stmt in valid.split_uop(Ops.AND):
    if (res:=parse_valid(stmt)) is None: continue
    X, is_upper_bound, c = res

    # for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
    if not is_upper_bound and c == 1 and all(u.op in GroupOp.Irreducible and u.vmin == 0 for u in X.split_uop(Ops.ADD)):
      testidx = functools.reduce(lambda nowidx,u: nowidx.substitute({u:u.const_like(0)}), X.split_uop(Ops.ADD), idx)
      if testidx.gep(0).vmax < 0 or testidx.gep(1).vmax < 0:
        drop_stmt.append(stmt)
        continue

    # if X <= c, check if it's out of bound when X = c+1
    # if X >= c, check if it's out of bound when X = c-1
    test_value = c + 1 if is_upper_bound else c - 1
    for i,b in zip(idx.src, (width, height)):
      if i.is_increasing():
        rw = i.substitute({X:X.const_like(test_value)})
        if rw.vmin >= b or rw.vmax < 0:
          drop_stmt.append(stmt)
          break
  return drop_stmt

def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> UOp|None:
  idx = uop_given_valid(valid, start_idx)
  return None if idx is start_idx else buf.index(idx.valid(valid), ptr=True)

def simplify_valid_image_load(buf:UOp, idx_y:UOp, idx_x:UOp, valid:UOp) -> UOp|None:
  if not isinstance(buf.dtype, ImageDType): return None
  start_idx = UOp.vectorize(idx_x, idx_y)
  idx = uop_given_valid(valid, start_idx)
  drop_stmt = _drop_valid_stmts(valid, idx, buf.dtype.shape[0], buf.dtype.shape[1])

  if not drop_stmt and idx is start_idx: return None
  new_valid = UOp.uprod(*ss) if (ss:=[s for s in valid.split_uop(Ops.AND) if s not in drop_stmt]) else None
  idx_y, idx_x = idx.gep(1), idx.gep(0)
  return buf.index(idx_y.valid(new_valid), idx_x.valid(new_valid), ptr=True) if new_valid is not None else buf.index(idx_y, idx_x, ptr=True)

load_store_indexing = PatternMatcher([
  # image load valid idx simplification
  (UPat(Ops.INDEX, src=(UPat.var("buf"), invalid_gate)), lambda buf,x,i,cond: simplify_valid_load(buf, x, cond)),
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("valid").where(UPat.var("idx_y"), UPat(arg=Invalid)),
                                         UPat.var("valid").where(UPat.var("idx_x"), UPat(arg=Invalid)))), simplify_valid_image_load),
])

# ***** load/store grouping *****

def expand_index(ctx, buf:UOp, vec:UOp):
  # determine optimal image shapes
  if isinstance(dt:=buf.dtype, ImageDType):
    x, valid = vec.get_idx().gep(0), vec.get_valid().gep(0)
    # search for dims that drop the most valid statements
    best_drop, cands = -1, []
    for ch, cw in ImageDType.valid_dims(dt, ctx.target.arch):
      if (dropped:=len(_drop_valid_stmts(valid, cidx:=uop_given_valid(valid, UOp.vectorize((x//4)%cw, x//(4*cw))), ch, cw))) > best_drop:
        best_drop, cands = dropped, [(ch, cw, cidx)]
      elif dropped == best_drop: cands.append((ch, cw, cidx))
    # and tiebreak with indexing complexity (ie. number of nodes)
    h, w, _ = cands[0] if len(cands) == 1 else min(cands, key=lambda cand: len(cand[2].gep(1).simplify().backward_slice))
    assert buf.op is Ops.RESHAPE
    buf = buf.src[0].replace(dtype=(dtypes.imageh if dt.itemsize == 2 else dtypes.imagef)((h, w, 4))).flatten()
  if getenv("UNSAFE_DISABLE_MASK", 0): vec = vec.get_idx()
  # generate the individual indexes
  return UOp(Ops.STACK, buf.dtype, tuple(buf.index(vec.gep(i), ptr=True) for i in range(vec.dtype.count)))

def fold_expanded_index(midx:UOp):
  buf = midx.src[0].src[0]
  if not all(s.src[0] is buf for s in midx.src): return None
  if not all(isinstance(s.dtype, PtrDType) for s in midx.src): return None

  # extract all the relevant offsets
  offsets_rootsrc: defaultdict[Any, dict[int, list[int]]] = defaultdict(dict)
  for i in range(len(midx.src)):
    idx: Any = midx.src[i].src[1].get_idx()
    if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST: root_src, arg = idx.src[0], idx.src[1].arg
    elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST: root_src, arg = idx.src[1], idx.src[0].arg
    elif idx.op is Ops.CONST and idx.arg is Invalid: root_src, arg = "INVALID", 0
    elif idx.op is Ops.CONST: root_src, arg = "CONST", idx.arg
    else: root_src, arg = idx, 0
    root_src = (midx.src[i].src[1].get_valid(), root_src)
    offsets_rootsrc[root_src].setdefault(arg, []).append(i)

  # then rewrite everything we can into groups
  ret = []
  idxs: list[int|None] = [None]*len(midx.src)
  global_offset = 0
  for offsets in offsets_rootsrc.values():
    grouped_offsets = [[x for _,x in group] for _,group in itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])]
    for grp in grouped_offsets:
      # get the index offset for this element. using [0] is okay, because they are the same
      lidx = midx.src[offsets[grp[0]][0]]
      if len(grp) > 1: lidx = lidx.cast(buf.ptrdtype.base.vec(len(grp)).ptr(size=buf.ptrdtype.size, addrspace=buf.ptrdtype.addrspace))
      # set the idxs of the output
      for i,g in enumerate(grp):
        for oo in offsets[g]: idxs[oo] = global_offset+i
      # add this lidx to the CAT
      ret.append(lidx)
      global_offset += len(grp)
  assert None not in idxs, f"some idxs are missing {idxs}"
  # this base thing is for image, we want the CAT to be a normal pointer
  post_cat = UOp(Ops.PTRCAT, buf.ptrdtype.base.ptr(size=buf.ptrdtype.size, addrspace=buf.ptrdtype.addrspace).vec(global_offset), tuple(ret))
  return post_cat.gep(tuple(cast(list[int], idxs)))

def cat_after_store(cat:UOp, data:UOp):
  # TODO: this is written in many places
  offset = 0
  ret: list[UOp] = []
  for s in cat.src:
    ret.append(s.store(data.gep(tuple(range(offset, offset+s.dtype.count)))))
    offset += s.dtype.count
  return UOp.group(*ret)

def gep_on_store(gep:UOp, st:UOp):
  # NOTE: we need to invert the gep here, but it may be an expanding gep
  # fake argsort. TODO: handle duplicates
  a = {}
  for i,x in enumerate(gep.arg): a[x] = i
  new_arg = tuple(x[1] for x in sorted(a.items()))
  return gep.src[0].store(st.gep(new_arg))

load_store_folding = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, src=UPat(name="buf")), UPat.var("vec"))), expand_index),
  (UPat(Ops.STACK, src=UPat(Ops.INDEX), name="midx"), fold_expanded_index),
  # GEP after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.GEP, name="gep"),), name="ld", allow_any_len=True),
   lambda gep, ld: ld.replace(dtype=ld.dtype.scalar().vec(gep.dtype.count), src=(gep.src[0],)+ld.src[1:]).gep(gep.arg)),
  # GEP on data of STORE
  (UPat(Ops.STORE, src=(UPat(Ops.GEP, name="gep"), UPat.var("st"))), gep_on_store),
  # put PTRCAT after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.PTRCAT, name="cat"),), name="ld", allow_any_len=True),
   lambda cat,ld: UOp(Ops.VCAT, cat.dtype.base.vec(cat.dtype.vcount), tuple(ld.replace(dtype=x.dtype.base, src=(x,)+ld.src[1:]) for x in cat.src))),
  # put PTRCAT after STORE
  (UPat(Ops.STORE, src=(UPat(Ops.PTRCAT, name="cat"), UPat(name="data"))), cat_after_store),
])

# *** correct load/store ***

def split_load_store(ctx:Renderer|None, ls:UOp, idx:UOp):
  # this splits loads and stores into multiple chunks

  # if there's only one element to load/store, no splitting needed
  if (sz:=ls.src[0].dtype.count) == 1: return None
  buf = idx.src[0]

  # determine fold lengths
  lengths = []
  must_divide = True
  if ctx is not None and ctx.target.device == "DSP":
    lengths = [128,64,32,16,8,4]
    must_divide = False
  elif buf.dtype.base not in (dtypes.float, dtypes.half, *dtypes.fp8s) and not isinstance(buf.dtype, ImageDType):
    pass
  elif buf.ptrdtype.addrspace == AddrSpace.REG:
    pass
  elif isinstance(buf.dtype, ImageDType):
    lengths = [4]
  elif ctx is not None and ctx.supports_float4:
    # TODO: a better way to get this than ctx
    lengths = [8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else ([16,8,4,2] if "AMX" in ctx.target.arch else [4,2])
  lengths.append(1)  # worst case, it's not folded

  # filter fold lengths that don't divide
  offset, mask = idx.src[1].get_idx(), idx.src[1].get_valid()
  if must_divide: lengths = [x for x in lengths if offset.divides(x) is not None]

  # split based on the fold lengths
  global_offset = 0
  ret = []
  while global_offset < sz:
    # with 1 at the end of the lengths list, this will always hit
    for fold_length in lengths:
      if global_offset+fold_length > sz: continue
      lidx = buf.index((offset + global_offset).valid(mask), ptr=True)
      if fold_length > 1: lidx = lidx.cast(buf.ptrdtype.base.vec(fold_length).ptr(size=buf.ptrdtype.size, addrspace=buf.ptrdtype.addrspace))
      if ls.op is Ops.STORE: ret.append(ls.replace(src=(lidx,ls.src[1].gep(tuple(range(global_offset, global_offset+fold_length))))))
      else: ret.append(ls.replace(src=(lidx,)+ls.src[1:], dtype=ls.dtype.scalar().vec(fold_length)))
      global_offset += fold_length
      break

  # if it wasn't split, we return None. otherwise we CAT them
  if len(ret) <= 1: return None
  return UOp(Ops.VCAT, ls.dtype, tuple(ret)) if ls.op is Ops.LOAD else UOp.group(*ret)

def get_image_idx(idx:UOp, width:int):
  x, valid = idx.src[1].get_idx(), idx.src[1].get_valid()
  idx_x, idx_y = (x // 4) % width, x // (4*width)
  assert idx.src[0].op is Ops.RESHAPE, "image idx must be on reshape"
  return idx.replace(src=(idx.src[0].src[0], idx_y.valid(valid), idx_x.valid(valid)))

def image_fixup(ls:UOp):
  # normal image load or store, with the CAST from expand_index
  if isinstance(dt:=ls.src[0].src[0].dtype, ImageDType) and ls.src[0].op is Ops.CAST:
    assert ls.src[0].dtype.count == 4, "image must be casted to 4"
    return ls.replace(src=(get_image_idx(ls.src[0].src[0], dt.shape[1]),)+ls.src[1:])

  # this is an unprocessed image without a cast, we should just make it a buffer
  if isinstance(dt, ImageDType) and len(ls.src[0].src) == 2:
    off = ls.src[0].src[1]
    assert ls.src[0].src[0].op is Ops.RESHAPE, "image idx must be on reshape"
    idx = ls.src[0].src[0].src[0].replace(dtype=(new_dt:=dtypes.half if dt.itemsize == 2 else dtypes.float).ptr(dt.size)).index(off)
    return ls.replace(src=(idx,), dtype=new_dt).cast(dtypes.float) if ls.op is Ops.LOAD else ls.replace(src=(idx, ls.src[1].cast(new_dt)))

correct_load_store = PatternMatcher([
  # split LOAD/STORE
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.INDEX, name="idx").cast(),), name="ls", allow_any_len=True), split_load_store),
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
  return UOp(Ops.STACK, wmma.dtype, tuple(wmma_ex))

def no_vectorized_alu(alu:UOp):
  if alu.dtype.vcount == 1: return None
  if alu.op is Ops.WHERE and alu.src[2].arg is Invalid: return None  # image load/store has cond.where(idx.vec(2), Invalid) as the index
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.STACK, alu.dtype, alus)

def no_vectorized_buf(buf:UOp):
  return buf.replace(dtype=buf.ptrdtype.base.scalar().ptr(buf.ptrdtype.size*buf.ptrdtype.count, buf.ptrdtype.addrspace)).cast(buf.dtype)

def no_vectorized_index(buf:UOp, cast:UOp, idx:UOp, bcast:UOp|None=None):
  cnt = cast.dtype.count
  if bcast is not None and bcast.op is Ops.GEP:
    # GEP selects specific lanes; bcast.arg[k] is the offset for lane k, iterate groups × selected lanes
    pairs = [(k, g + bcast.arg[k]) for g, k in itertools.product(range(cast.dtype.vcount), range(len(bcast.arg)))]
  elif bcast is not None:
    # BROADCAST: cross product of components × lanes
    pairs = [(j, c) for c, j in itertools.product(range(cnt), range(bcast.dtype.vcount))]
  else:
    # simple scalar index: one lane, all components
    pairs = [(0, c) for c in range(cnt)]
  idx_lanes, offsets = (tuple(x) for x in zip(*pairs))
  return buf.broadcast(len(pairs)).index(idx.gep(idx_lanes)*cnt + UOp.const(dtypes.weakint.vec(len(pairs)), offsets), ptr=True)

devectorize_buf_and_index = PatternMatcher([
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="buf"), no_vectorized_buf),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").index(UPat.var("idx")), no_vectorized_index),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").broadcast(name="bcast").index(UPat.var("idx")),
   no_vectorized_index),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").gep(name="bcast").index(UPat.var("idx")),
   no_vectorized_index),
])

devectorize_alu = PatternMatcher([
  # CAST after AFTER
  (UPat(Ops.CAST, name="c").f(Ops.AFTER, allow_any_len=True, name="a"), lambda c,a: c.src[0].after(*a.src[1:]).cast(c.dtype)),
  # no ALU on vectorized dtypes
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
])

pm_render = PatternMatcher([
  # for rendering, we use explicit VECTORIZE
  (UPat(Ops.CONST, name='c'),
   lambda c: UOp(Ops.STACK, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.STACK, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  (UPat(Ops.GEP, name='gep'), lambda gep: gep.src[0] if gep.src[0].dtype.vcount == 1 and gep.arg == (0,) else None),
  (UPat(Ops.STACK, src=(UPat(name='x'),)), lambda x: x),
])

# *** Ops.REDUCE -> Ops.DEFINE_ACC ***

@dataclass
class ReduceContext:
  acc_num: int = 0

def horizontal_reduce(inp:UOp, out_dtype:DType) -> list[UOp]:
  # if this has a horizontal reduction component, do that first
  if inp.dtype != out_dtype:
    # NOTE: [0 1 2 3 4 5 6 7] -> [0+4, 1+5, 2+6, 3+7]
    horizontal_amount = inp.dtype.count//out_dtype.count
    return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(0, horizontal_amount)]
  return [inp]

def reduce_to_acc(ctx:ReduceContext, red:UOp):
  inp, reduce_range = red.src[0], red.src[1:]
  lst = horizontal_reduce(inp, red.dtype)
  assert all(x.dtype == red.dtype for x in lst), f"horizontal reduction mismatch {lst[0].dtype} != {red.dtype}"
  # if we have a range
  if len(reduce_range) != 0:
    topo = inp.toposort()
    ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
    input_ranges = tuple([x for x in topo if x.op is Ops.RANGE and x not in reduce_range and x not in ended_ranges])
    identity = red.const(red.dtype, identity_element(red.arg[0], red.dtype.scalar()))
    acc = UOp.placeholder((1,), red.dtype, ctx.acc_num, AddrSpace.REG)
    acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(identity)
    lst = [acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))] + lst  # put acc as the first element
    ctx.acc_num += 1
  ret = functools.reduce(lambda x,y: x.alu(red.arg[0], y), lst)
  if len(reduce_range) == 0: return ret
  end = acc.index(UOp.const(dtypes.weakint, 0)).store(ret).end(*reduce_range).rtag("mergeable")
  return acc.after(end).index(UOp.const(dtypes.weakint, 0))

def merge_reduce_ends(ctx:ReduceContext, sink:UOp):
  # merge ENDs that share the same range and nesting context (only those created by reduce_to_acc)
  # ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END
  range_to_ends: dict[tuple[UOp, ...], list[UOp]] = {}
  for u in sink.backward_slice:
    if u.op is Ops.END and u.tag == "mergeable": range_to_ends.setdefault(u.src[1:], []).append(u)
  subs: dict[UOp, UOp] = {}
  next_axis = max((u.arg[0] for u in sink.backward_slice if u.op is Ops.RANGE), default=-1) + 1
  for r, ends in range_to_ends.items():
    if len(ends) <= 1: continue
    by_ctx: dict[frozenset[UOp], list[UOp]] = {}
    for e in ends: by_ctx.setdefault(frozenset(e.ranges), []).append(e)
    for i, group in enumerate(by_ctx.values()):
      tr = r if i == 0 else tuple(rr.replace(arg=(next_axis + j, *rr.arg[1:])) for j, rr in enumerate(r))
      if i > 0: next_axis += len(r)
      mapped = [e.substitute(dict(zip(r, tr))) if i > 0 else e for e in group]
      merged = mapped[0] if len(mapped) == 1 else UOp.group(*(e.src[0] for e in mapped)).end(*tr)
      for e in group: subs[e] = merged
  return sink.substitute(subs) if subs else None

pm_reduce = PatternMatcher([
  # REDUCE -> DEFINE_ACC+ASSIGN, then merge ENDs with same range
  (UPat(Ops.REDUCE, name="red"), reduce_to_acc),
  (UPat(Ops.SINK, name="sink"), merge_reduce_ends),
  # tensor core built in accumulate
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
])

# add loads

def add_load(idx:UOp):
  if isinstance(idx.dtype, PtrDType): return None
  assert isinstance(idx.src[0].dtype, PtrDType), f"param is not PtrDType {idx.src[0].dtype}"
  return idx.replace(dtype=idx.src[0].dtype).load(dtype=idx.dtype.base)

pm_add_loads = PatternMatcher([
  # add loads to non ptr index
  (UPat(Ops.INDEX, name="idx"), add_load),
  # remove loads from stores
  (UPat(Ops.STORE, src=(UPat(Ops.LOAD),), allow_any_len=True, name="s"), lambda s: s.replace(src=(s.src[0].src[0],)+s.src[1:])),
  (UPat(Ops.LOAD, src=(UPat(Ops.LOAD),), allow_any_len=True, name="l"), lambda l: l.replace(src=(l.src[0].src[0],)+l.src[1:])),
])

# make images

pm_imageh_store = PatternMatcher([
  # store<imageh>(idx, x) is actually store(idx, x.cast(half)) so we can pull the cast into the store
  (UPat.var("x", dtypes.float).cast(dtypes.half), lambda x: x),
  # store(imageh, a.where(b.half(), c).float()) -> store(imageh, a.where(b, c.float()))
  (UPat(Ops.WHERE, src=(UPat.var("a"), UPat.var("b", dtypes.float).cast(dtypes.half), UPat.var("c"))), lambda a,b,c: a.where(b,c.cast(dtypes.float))),
  # otherwise, we cast to float
  (UPat(GroupOp.All, name="x"), lambda x: x.cast(dtypes.float))
])

def make_image(ctx, ls, buf, off):
  if (vcount:=buf.dtype.vcount) != 1: buf = buf.src[0]
  if buf.op == Ops.PARAM and not isinstance(dt:=buf.dtype, ImageDType) and (dims:=ImageDType.valid_dims(dt, ctx)):
    buf = buf.replace(dtype=(dtypes.imageh if dt.base == dtypes.half else dtypes.imagef)((*dims[0], 4))).flatten()
    if vcount != 1: buf = UOp.vectorize(*([buf] * vcount))
    if ls.op is Ops.LOAD: return ls.replace(src=(buf.index(off, ptr=True),), dtype=dtypes.float.vec(ls.dtype.vcount)).cast(dt.base)
    return buf.index(off, ptr=True).store(pm_imageh_store.rewrite(ls.src[1]) if dt.base == dtypes.half else ls.src[1])

pm_make_images = PatternMatcher([
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off"))),), allow_any_len=True, name="ls"), make_image),
  # load<imageh> is actually load<half>.cast(float), so load<imageh>.half().float() -> load<half>.float().half().float() -> load<half>.float()
  (UPat(Ops.LOAD, name="li").cast(dtypes.half).cast(dtypes.float), lambda li: li if isinstance(li.src[0].dtype, ImageDType) else None),
])
