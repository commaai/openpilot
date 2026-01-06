from typing import Any, cast
import functools, operator, itertools
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.dtype import dtypes, ImageDType, DType, AddrSpace, Invalid, PtrDType
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp, identity_element
from tinygrad.uop.symbolic import uop_given_valid, parse_valid, invalid_gate
from tinygrad.helpers import getenv, flatten, AMX, prod
from tinygrad.renderer import Renderer

# ***** image load valid simplification *****

def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> UOp|None:
  idx = uop_given_valid(valid, start_idx)
  if not isinstance(buf.dtype, ImageDType): return None if idx is start_idx else buf.index(idx.valid(valid), ptr=True)

  # wait for it to be image indexed before running simplification
  if start_idx.dtype.count != 2: return None

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
    for i,b in zip(idx.src, (buf.dtype.shape[1], buf.dtype.shape[0])):
      if i.is_increasing():
        rw = i.substitute({X:X.const_like(test_value)})
        if rw.vmin >= b or rw.vmax < 0:
          drop_stmt.append(stmt)
          break

  if not drop_stmt and idx is start_idx: return None
  new_valid = functools.reduce(operator.and_, ss) if (ss:=[s for s in valid.split_uop(Ops.AND) if s not in drop_stmt]) else None
  return buf.index(idx.valid(new_valid) if new_valid is not None else idx, ptr=True)


load_store_indexing = PatternMatcher([
  # image load valid idx simplification
  (UPat(Ops.INDEX, src=(UPat.var("buf"), invalid_gate)), lambda buf,x,i,cond: simplify_valid_load(buf, x, cond)),
  # simplify away long after index has been lowered
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("x", dtypes.long), UPat.var("c", dtypes.bool))), lambda buf,x,c: simplify_valid_load(buf, x, c)),
  # drop true gate
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("x"), UPat.const(dtypes.bool, True)),), lambda buf,x: buf.index(x, ptr=True)),
])

# ***** load/store grouping *****

def expand_index(buf:UOp, vec:UOp):
  if getenv("UNSAFE_DISABLE_MASK", 0): vec = vec.get_idx()
  # generate the individual indexes
  return UOp(Ops.VECTORIZE, buf.dtype, tuple(buf.index(vec.gep(i), ptr=True) for i in range(vec.dtype.count)))

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

def cat_after_store(cat:UOp, data:UOp, sto:UOp):
  # TODO: this is written in many places
  offset = 0
  ret: list[UOp] = []
  for s in cat.src:
    ret.append(s.store(data.gep(tuple(range(offset, offset+s.dtype.count))), *sto.src[2:]))
    offset += s.dtype.count
  return UOp.group(*ret)

def gep_on_store(gep:UOp, st:UOp, sto:UOp):
  # NOTE: we need to invert the gep here, but it may be an expanding gep
  # fake argsort. TODO: handle duplicates
  a = {}
  for i,x in enumerate(gep.arg): a[x] = i
  new_arg = tuple(x[1] for x in sorted(a.items()))
  return gep.src[0].store(st.gep(new_arg), *sto.src[2:])

load_store_folding = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat(Ops.VECTORIZE, src=UPat(GroupOp.Defines).or_after(name="buf")), UPat.var("vec"))), expand_index),
  (UPat(Ops.VECTORIZE, src=UPat(Ops.INDEX), name="midx"), fold_expanded_index),
  # GEP after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.GEP, name="gep"),), name="ld", allow_any_len=True),
   lambda gep, ld: ld.replace(dtype=ld.dtype.scalar().vec(gep.dtype.count), src=(gep.src[0],)+ld.src[1:]).gep(gep.arg)),
  # GEP on data of STORE
  (UPat(Ops.STORE, src=(UPat(Ops.GEP, name="gep"), UPat.var("st")), allow_any_len=True, name="sto"), gep_on_store),
  # put PTRCAT after LOAD
  (UPat(Ops.LOAD, src=(UPat(Ops.PTRCAT, name="cat"),), name="ld", allow_any_len=True),
   lambda cat,ld: UOp(Ops.CAT, cat.dtype.base.vec(cat.dtype.vcount), tuple(ld.replace(dtype=x.dtype.base, src=(x,)+ld.src[1:]) for x in cat.src))),
  # put PTRCAT after STORE
  (UPat(Ops.STORE, src=(UPat(Ops.PTRCAT, name="cat"), UPat(name="data")), allow_any_len=True, name="sto"), cat_after_store),
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
  if ctx is not None and ctx.device == "DSP":
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
    lengths = [8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else ([16,8,4,2] if AMX else [4,2])
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
      if ls.op is Ops.STORE: ret.append(ls.replace(src=(lidx,ls.src[1].gep(tuple(range(global_offset, global_offset+fold_length))))+ls.src[2:]))
      else: ret.append(ls.replace(src=(lidx,)+ls.src[1:], dtype=ls.dtype.scalar().vec(fold_length)))
      global_offset += fold_length
      break

  # if it wasn't split, we return None. otherwise we CAT them
  if len(ret) <= 1: return None
  return UOp(Ops.CAT, ls.dtype, tuple(ret)) if ls.op is Ops.LOAD else UOp.group(*ret)

def image_fixup(ls:UOp):
  # normal image load or store, with the CAST from expand_index
  if ls.src[0].op is Ops.CAST and isinstance(image_dtype:=ls.src[0].src[0].dtype, ImageDType):
    assert ls.src[0].dtype.count == 4, "image must be casted to 4"
    idx = ls.src[0].src[0]
    x, valid = idx.src[1].get_idx(), idx.src[1].get_valid()
    oidx = UOp(Ops.VECTORIZE, dtypes.index.vec(2), ((x // 4) % image_dtype.shape[1], (x // (4*image_dtype.shape[1]))))
    idx = idx.replace(src=(idx.src[0], oidx.valid(valid)))
    return ls.replace(src=(idx,)+ls.src[1:])

  # this is an unprocessed image without a cast, aka unfoldable image load. this doesn't work for stores
  if isinstance(image_dtype:=ls.src[0].dtype, ImageDType) and ls.src[0].src[1].get_idx().dtype != dtypes.index.vec(2):
    assert ls.op is Ops.LOAD, "if an image store isn't upcasted to 4, we can't store it"
    idx = ls.src[0]
    x, valid = idx.src[1].get_idx(), idx.src[1].get_valid()
    oidx = UOp(Ops.VECTORIZE, dtypes.index.vec(2), ((x // 4) % image_dtype.shape[1], (x // (4*image_dtype.shape[1]))))
    idx = idx.replace(src=(idx.src[0], oidx.valid(valid)))
    vec_load = ls.replace(dtype=ls.dtype.vec(4), src=(idx,)+ls.src[1:])
    return functools.reduce(lambda ret, i: (x % 4).ne(i).where(ret, vec_load.gep(i)), range(4), ls.const_like(float('nan')))

  return None

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
  return UOp(Ops.VECTORIZE, wmma.dtype, tuple(wmma_ex))

def no_vectorized_alu(alu:UOp):
  if alu.dtype.vcount == 1: return None
  if alu.op is Ops.WHERE and alu.src[2].arg is Invalid: return None  # image load/store has cond.where(idx.vec(2), Invalid) as the index
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.VECTORIZE, alu.dtype, alus)

def no_vectorized_buf(buf:UOp):
  return buf.replace(dtype=buf.ptrdtype.base.scalar().ptr(buf.ptrdtype.size*buf.ptrdtype.count, buf.ptrdtype.addrspace)).cast(buf.dtype)

def no_vectorized_index(buf:UOp, cast:UOp, idx:UOp):
  cnt = cast.dtype.count
  assert idx.dtype.count == 1, f"idx dtype must be 1 {idx.dtype}"
  return buf.broadcast(cnt).index(idx.broadcast(cnt)*cnt+UOp.const(dtypes.index.vec(cnt), tuple(range(cnt))), ptr=True)

def no_vectorized_index_broadcast(buf:UOp, cast:UOp, bcast:UOp, idx:UOp):
  cnt = cast.dtype.count
  precnt = bcast.dtype.vcount
  input_gep = bcast.arg if bcast.op is Ops.GEP else ([0]*precnt)
  gep_arg = tuple(flatten([range(precnt) for _ in range(cnt)]))
  sum_arg = tuple(flatten([[i+y for y in input_gep] for i in range(cnt)]))
  return buf.broadcast(cnt*precnt).index(idx.gep(gep_arg)*cnt+UOp.const(dtypes.index.vec(cnt*precnt), sum_arg), ptr=True)

devectorize_buf_and_index = PatternMatcher([
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="buf"), no_vectorized_buf),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").index(UPat.var("idx")), no_vectorized_index),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").broadcast(name="bcast").index(UPat.var("idx")),
   no_vectorized_index_broadcast),
  (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").gep(name="bcast").index(UPat.var("idx")),
   no_vectorized_index_broadcast),
])

devectorize = PatternMatcher([
  # CAST after AFTER
  (UPat(Ops.CAST, name="c").f(Ops.AFTER, allow_any_len=True, name="a"), lambda c,a: c.src[0].after(*a.src[1:]).cast(c.dtype)),
  # no ALU on vectorized dtypes
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
])+devectorize_buf_and_index

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
    lambda x: x.replace(src=(x.src[0], x.const_like(0))+x.src[1:])
      if len(x.src) == 1 or x.src[1].op in (Ops.CUSTOM, Ops.STORE, Ops.BARRIER) else None),
  # Where after gated load becomes alt value
  (UPat.var("c").where(UPat(Ops.LOAD, src=(UPat().index(UPat.var("idx"), UPat.var("c")).or_casted(),), allow_any_len=True, name="l").or_casted(),
    UPat.var("a")), lambda c,idx,l,a: l.replace(src=(l.src[0], a.cast(l.dtype))+l.src[2:]).cast(a.dtype)),
  (UPat.var("c").where(UPat.var("a"), UPat(Ops.LOAD, src=(UPat().index(UPat.var("idx"), UPat.var("c").logical_not()).or_casted(),),
    allow_any_len=True, name="l").or_casted()), lambda c,idx,l,a: l.replace(src=(l.src[0], a.cast(l.dtype))+l.src[2:]).cast(a.dtype)),
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
    identity = red.const(red.dtype, identity_element(red.arg, red.dtype.scalar()))
    acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=ctx.acc_num)
    acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.int, 0)).store(identity) if len(input_ranges) else \
               acc.index(UOp.const(dtypes.int, 0)).store(identity)
    lst = [acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.int, 0))] + lst  # put acc as the first element
    ctx.acc_num += 1
  ret = functools.reduce(lambda x,y: x.alu(red.arg, y), lst)
  if len(reduce_range) == 0: return ret
  return acc.after(acc.index(UOp.const(dtypes.int, 0)).store(ret).end(*reduce_range)).index(UOp.const(dtypes.int, 0))

pm_reduce = PatternMatcher([
  # REDUCE -> DEFINE_ACC+ASSIGN
  (UPat(Ops.REDUCE, name="red"), reduce_to_acc),
  # tensor core built in accumulate
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
])

# add loads

pm_add_loads = PatternMatcher([
  # add loads to non ptr index
  (UPat(Ops.INDEX, name="idx"), lambda idx: None if isinstance(idx.dtype, (PtrDType, ImageDType)) else
    idx.replace(dtype=idx.src[0].dtype).load(dtype=idx.dtype.base)),
  # remove loads from stores
  (UPat(Ops.STORE, src=(UPat(Ops.LOAD),), allow_any_len=True, name="s"), lambda s: s.replace(src=(s.src[0].src[0],)+s.src[1:])),
])

