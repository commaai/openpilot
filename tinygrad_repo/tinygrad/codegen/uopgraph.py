from __future__ import annotations
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING, Any, DefaultDict, Callable, Set
import functools, itertools, operator
from collections import defaultdict
from tinygrad.dtype import dtypes, ImageDType, PtrDType
from tinygrad.ops import UOp, Ops, UPat, PatternMatcher, symbolic_flat, symbolic_simple
from tinygrad.ops import graph_rewrite, split_uop, uop_given_valid, parse_valid, is_increasing, simplify_valid, GroupOp
from tinygrad.helpers import DEBUG, getenv, flatten, dedup, TRANSCENDENTAL, AMX, prod, partition, all_same
from tinygrad.codegen.transcendental import xexp2, xlog2, xsin, TRANSCENDENTAL_SUPPORTED_DTYPES

if TYPE_CHECKING: from tinygrad.renderer import Renderer

# ***** float4/image store handling *****

def fold_expanded(ex, buf):
  if buf.dtype.base != dtypes.float and buf.dtype.base != dtypes.half and not isinstance(buf.dtype, ImageDType): return None
  new_srcs = dedup(list(ex.src))
  old_new_srcs = new_srcs[:]
  is_load, is_image = new_srcs[0].op is Ops.LOAD, isinstance(buf.dtype, ImageDType)

  # first, extract all the relevant offsets
  offsets_rootsrc: DefaultDict[Any, dict] = defaultdict(dict)
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
  lengths = [4] if is_image else ([8,4,2] if buf.dtype.base == dtypes.half and getenv("ALLOW_HALF8") else ([16,8,4,2] if AMX else [4,2]))
  used: Set[Tuple[UOp, UOp]] = set()
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
            new_src[0] = new_src[0].cast(new_src[0].dtype.base.vec(fold_length).ptr(new_src[0].dtype.local))
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

def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> Optional[UOp]:
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
      if is_increasing(i):
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
  pat: List[Tuple[UPat, Callable]] = [(UPat(op, dtype=TRANSCENDENTAL_SUPPORTED_DTYPES, src=(UPat.var("d"),)), f) for op,f in \
           ((Ops.EXP2, xexp2), (Ops.LOG2, xlog2), (Ops.SIN, xsin)) if op not in ops or force_transcendental]
  # rewrite MOD to AND (which should always be supported, but not for generic in tests): x % (2**y) -> x & (2**y-1)
  if Ops.AND in ops:
    pat += [(UPat.var("x", dtypes.ints)%UPat.cvar("c"), lambda x,c: x & (c.arg-1) if c.arg in powers_of_two else None)]
  # rewrite MUL/IDIV to SHL+SHR: x*(2**y) -> shl(x,y) and x//(2**y) -> shr(x,y)
  if Ops.SHL in ops and Ops.SHR in ops:
    pat += [
      (UPat.var("x", dtypes.ints)*UPat.cvar("c"), lambda c,x: x << powers_of_two[c.arg] if c.arg in powers_of_two else None),
      (UPat.var("x", dtypes.ints)//UPat.cvar("c"), lambda x,c: x >> powers_of_two[c.arg] if c.arg in powers_of_two else None)
    ]
  if Ops.NEG in ops:
    pat += [(UPat.var('x')*-1, lambda x: x.alu(Ops.NEG))]
    if Ops.SUB in ops: pat += [(UPat.var('x')+UPat.var('y').alu(Ops.NEG), lambda x,y: x.alu(Ops.SUB, y))]
  if Ops.MULACC in ops:
    pat += [(UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c))]
  return PatternMatcher(pat)

# ***** threefry *****

def threefry2x32(x: UOp, key: UOp):
  # split x into two uint32, since x in a uint64
  x0, x1 = (x & 0xffffffff).cast(dtypes.uint32), ((x // 2**32) & 0xffffffff).cast(dtypes.uint32)

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  key0, key1 = (key & 0xffffffff).cast(dtypes.uint32), ((key // 2**32) & 0xffffffff).cast(dtypes.uint32)
  ks = [key1, key0 ^ key1 ^ 0x1BD11BDA, key0]
  xr = [x0 + ks[-1], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]: xr[0], xr[1] = (x0 := xr[0] + xr[1]), x0 ^ ((xr[1] * 2**r) + (xr[1] // 2**(32 - r)))
    xr = [(xr[0] + ks[i % 3]), (xr[1] + ks[(i + 1) % 3] + i + 1)]

  return xr[1].cast(dtypes.uint64) * 2**32 | xr[0].cast(dtypes.uint64)

# ***** other math rewrite ****

def sigmoid_like(x:UOp, y:UOp): return (t:=(1/(x+1))) * (1-t) * y

# ***** main rewriter *****

def loop_collapse(compval, multconst, rng:UOp, acc:UOp, idx2=None,idx3=None,extra=None,vec=None,ne=None,
                  add=UOp.const(dtypes.int, 0), mul:UOp=UOp.const(dtypes.int, 1)):
  if getenv("DISABLE_LOOP_COLLAPSE") or rng not in acc.src: return None  # must be the right REDUCE
  loop_start, loop_end = rng.src
  if loop_start.arg != 0:
    # TODO: support and test this with other mul and loop_starts
    if DEBUG >= 1: print(f"WARNING, NOT FOLDING: mul:{mul.arg} loop_start:{loop_start.arg}")
    return None
  if idx2 is not None: add = add + idx2
  if idx3 is not None: add = add + idx3
  if vec is not None:
    # add, mul, loop_start, loop_end
    def dvec(x:UOp):
      if x.op is Ops.CONST: return UOp.const(x.dtype.vec(vec.dtype.count), x.arg)
      return UOp(Ops.VECTORIZE, x.dtype.vec(vec.dtype.count), src=(x,)*vec.dtype.count)
    add, mul, loop_start, loop_end = dvec(add), dvec(mul), dvec(loop_start), dvec(loop_end)
  if mul.vmin > 0 and ne is not None:
    comprange = UOp.minimum(loop_end, UOp.maximum((add-compval)//mul + (loop_end-loop_start), loop_start))
  elif mul.vmax < 0 and ne is None:
    comprange = UOp.minimum(loop_end, UOp.maximum((add-compval-mul)//mul + (loop_end-loop_start), loop_start))
  else:
    return None
  new_reduce_op = comprange.cast(multconst.dtype) * multconst
  # TODO: what does it mean to have the same numbered DEFINE_ACC with different ranges?
  new_acc = acc.replace(src=acc.src[0:1]+tuple(x for x in acc.src[1:] if x is not rng))
  ret = new_acc.assign(new_acc+new_reduce_op)
  if extra is not None: ret = ret + acc.assign(acc+extra)
  return ret

def index_collapse(idx:UOp,rng:UOp,buf:UOp,ld:UOp,acc:UOp,add=UOp.const(dtypes.int, 0),mul=UOp.const(dtypes.int, 1)):
  if rng not in acc.src: return None
  new_load = UOp.load(buf.index(add+mul*idx, (idx >= rng.src[0]) & (idx < rng.src[1])), dtype=ld.dtype)
  new_acc = acc.replace(src=acc.src[0:1]+tuple(x for x in acc.src[1:] if x is not rng))
  return new_acc.assign(new_acc+new_load)

# TODO: there's a lot shared with no_vectorized_wmma here
def gep_through_wmma(gep:UOp, wmma:UOp):
  out_sz = prod(x[1] for x in wmma.arg[6][-1])
  wmma_idxs = gep.arg[::out_sz]
  for i in range(out_sz):
    if tuple(x-i for x in gep.arg[i::out_sz]) != wmma_idxs: return None
  tsrcs = []
  for s,sz in zip(wmma.src, wmma.arg[6]):
    src_args = []
    ssz = prod(x[1] for x in sz)
    for w in wmma_idxs: src_args += list(range((w//out_sz)*ssz, (w//out_sz)*ssz + ssz))
    tsrcs.append(s.gep(tuple(src_args)))
  return UOp(Ops.WMMA, gep.dtype, tuple(tsrcs), wmma.arg)

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

def reduce_collapse(acc:UOp, ret:UOp, alu:UOp):
  reduce_parented, reduce_unparented = partition(acc.src[1:], lambda x: x in ret.toposort)
  if len(reduce_unparented) == 0: return None
  new_acc = acc.replace(src=acc.src[0:1]+tuple(reduce_parented))
  ret = new_acc.assign(new_acc.alu(alu.op, ret))
  if alu.op is Ops.ADD:
    for r in reduce_unparented: ret = ret * (r.src[1]-r.src[0]).cast(ret.dtype.scalar()).broadcast(ret.dtype.count)
  return ret

acc_pat, rng_pat = UPat(Ops.DEFINE_ACC, name="acc"), UPat(Ops.RANGE, name="rng")
rng_aug = UPat.any(rng_pat, UPat.var("add")+rng_pat, UPat.var("mul")*rng_pat, UPat.var("add")+UPat.var("mul")*rng_pat)

index_load = UPat.var("buf").index(rng_aug).load(name="ld")

arange_augrng = UPat.any(rng_aug, rng_aug+UPat.var("idx2"), rng_aug+UPat.var("idx2")+UPat.var("idx3"), UPat(Ops.VECTORIZE, name="vec", src=rng_aug))
arange_m = ((arange_augrng<UPat.cvar("compval"))!=UPat(Ops.CONST, name="ne", arg=True)).where(UPat.cvar("multconst"), UPat.const(None, 0))

# this is symbolic 2.0
sym = symbolic_flat+PatternMatcher([
  # self ASSIGN is just self
  (UPat(Ops.ASSIGN, src=(UPat.var('x'), UPat.var('x'))), lambda x: x),
  # VECTORIZE/CONST, VECTORIZE/GEP
  (UPat(Ops.VECTORIZE, src=UPat(Ops.CONST), name="vec"), lambda vec: UOp.const(vec.dtype, tuple(x.arg for x in vec.src))),
  (UPat(Ops.VECTORIZE, src=UPat(Ops.GEP, src=(UPat(name="x"),)), name="vec"), lambda vec,x: x.gep(tuple(y.arg[0] for y in vec.src))),
  # reorder ALU/VECTORIZE
  (UPat(GroupOp.ALU, src=(UPat(Ops.VECTORIZE, src=UPat(name='x')), UPat(Ops.VECTORIZE, src=UPat(name='y'))), name='alu'),
   lambda x,y,alu: UOp(Ops.VECTORIZE, alu.dtype, (UOp(alu.op, alu.dtype.scalar(), (x,y)),)*alu.dtype.count)),
  # VECTORIZE of a single element is just that element
  (UPat(Ops.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
  # VECTORIZE void is SINK
  (UPat(Ops.VECTORIZE, dtype=dtypes.void, src=UPat(Ops.BARRIER, name='b')), lambda b: b),
  (UPat(Ops.VECTORIZE, dtype=dtypes.void, name='x'), lambda x: UOp(Ops.SINK, dtypes.void, x.src)),
  # GEP/VECTORIZE, GEP/GEP, GEP/CONST, GEP/VCONST
  (UPat(Ops.GEP, src=(UPat(Ops.GEP, name='g2'),), name='g1'),
   lambda g1, g2: g2.src[0].gep(tuple(g2.arg[g1.arg[i]] for i in range(g1.dtype.count)))),
  (UPat(Ops.GEP, src=(UPat(Ops.VECTORIZE, name="vec"),), name="gep"),
   lambda gep, vec: UOp(Ops.VECTORIZE, gep.dtype, tuple(vec.src[i] for i in gep.arg)) if len(gep.arg) > 1 else vec.src[gep.arg[0]]),
  (UPat(Ops.GEP, src=(UPat.cvar("c", vec=False),), name="gep"), lambda gep, c: gep.const_like(c.arg)),
  (UPat(Ops.GEP, src=(UPat(Ops.VCONST, name="c"),), name="gep"), lambda gep, c: gep.const_like(tuple(c.arg[x] for x in gep.arg))),
  # push all GEPs through ALUs (fix arange stuff)
  (UPat(Ops.GEP, src=(UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name='alu'),), name='gep'),
   lambda gep,alu: UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count), tuple(x.gep(gep.arg) for x in alu.src), alu.arg)),
  # push some GEPs through WMMAs
  (UPat(Ops.GEP, src=(UPat(Ops.WMMA, name="wmma"),), name="gep"), gep_through_wmma),
  # tensor core with a 0 input is acc
  (UPat(Ops.WMMA, src=(UPat.const(None, 0.0), UPat.var(), UPat.var("acc"))), lambda acc: acc),
  (UPat(Ops.WMMA, src=(UPat.var(), UPat.const(None, 0.0), UPat.var("acc"))), lambda acc: acc),
  # tensor core cleanups
  (UPat.var("add") + UPat(Ops.WMMA, name="wmma"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
  # threefry + remove longs
  (UPat(Ops.THREEFRY, dtype=dtypes.uint64, src=(UPat.var("x"), UPat.var("key"))), threefry2x32),
  (UPat.var('x', dtypes.uint32).cast(dtypes.uint64).cast(dtypes.uint32), lambda x: x),   # cast there and back is noop (TODO: genericize)
  ((UPat.var('x', dtypes.uint64)&0xFFFFFFFF).cast(dtypes.uint32), lambda x: x.cast(dtypes.uint32)),  # cast does truncation
  (((UPat.var(None, dtypes.uint64)*(1<<32)) | UPat.var('y',  dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
  (((UPat.var('x',  dtypes.uint64)*(1<<32)) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))//(1<<32), lambda x: x),
  # hacks for threefry long removal when padded (TODO: genericize)
  (UPat.var('x', dtypes.uint32).cast(dtypes.uint64) * UPat.var('y').where(UPat.const(dtypes.uint64, 1<<32), UPat.const(dtypes.uint64, 0)),
   lambda x,y: y.where(x, UOp.const(dtypes.uint32, 0)).cast(dtypes.uint64) * (1<<32)),
  ((UPat.var('x', dtypes.uint64)&(UPat.var('y').where(UPat.const(dtypes.uint64, 0xFFFFFFFF), UPat.const(dtypes.uint64, 0)))).cast(dtypes.uint32),
   lambda x,y: y.where(x.cast(dtypes.uint32), UOp.const(dtypes.uint32, 0))),
  # arange loop folding
  (acc_pat.assign(UPat.any(arange_m, arange_m+UPat.var("extra"))+acc_pat), loop_collapse),
  # indexing, with cast or where
  (acc_pat.assign(UPat.var("idx").eq(UPat(Ops.RANGE, name="rng")).cast()*index_load+acc_pat), index_collapse),
  (acc_pat.assign(UPat.var("idx").eq(UPat(Ops.RANGE, name="rng")).where(index_load, UPat.const(None, 0.0))+acc_pat), index_collapse),
  # parentless reduce  # TODO: add MUL
  (acc_pat.assign(UPat((Ops.ADD, Ops.MAX), src=[acc_pat, UPat.var("ret")], name="alu")), reduce_collapse),
  # ** self folding **
  (UPat(Ops.DEFINE_ACC, src=(UPat.var("x"),)), lambda x: x),            # a DEFINE_ACC without ranges is a CONST
  (UPat(Ops.ASSIGN, src=(UPat.cvar(),UPat.var("x"))), lambda x: x),     # an ASSIGN to a const is a NOOP
  # x!=0 -> (bool)x
  (UPat.var("x")!=0, lambda x: x.cast(dtypes.bool.vec(x.dtype.count))),
  # ** load/store folding **
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.load(UPat(Ops.INDEX, name="index"))), lambda index: UOp(Ops.NOOP)),
  (UPat.store(UPat(Ops.INDEX, name="index"), UPat.var("gate").where(UPat.var("alt"), UPat.load(UPat(Ops.INDEX, name="index")))),
   lambda index, gate, alt: UOp.store(index.src[0].index(index.src[1], gate), alt)),
  # fold gated LOAD/STORE
  (UPat().index(UPat(), UPat.const(dtypes.bool, True)).named("idx"), lambda idx: idx.replace(src=idx.src[0:2])), # remove True
  (UPat().index(UPat(), UPat.const(dtypes.bool, False)).named("idx"), lambda idx: idx.const_like(0)),      # False -> NULL pointer
  (UPat(Ops.LOAD, src=(UPat.const(None, 0),), allow_any_len=True, name="x"), lambda x: x.const_like(0)),  # NULL pointer load loads 0
  (UPat(Ops.STORE, src=(UPat.const(None, 0),), allow_any_len=True), lambda: UOp(Ops.NOOP)),  # NULL pointer store does nothing
  # remove NOOPs from SINK
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(Ops.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not Ops.NOOP)) != len(root.src) else None),
  # remove VECTORIZE from SINK/BARRIER
  (UPat(Ops.BARRIER, src=(UPat((Ops.VECTORIZE, Ops.SINK), name='sink'),)), lambda sink: UOp(Ops.BARRIER, dtypes.void, sink.src)),
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(Ops.SINK, root.dtype, tuple(flatten(x.src if x.op in {Ops.SINK, Ops.UNROLL} else (x,) for x in root.src)), root.arg)
      if any(x.op in {Ops.SINK, Ops.UNROLL} for x in root.src) else None),
  # stable sigmoid
  (UPat.var("x")*(((UPat.var("x")+1)*(UPat.var("x")+1)).reciprocal()), lambda x: sigmoid_like(x, x.const_like(1))),
  (UPat.var("x")*(((UPat.var("x")+1)*(UPat.var("x")+1)).reciprocal()*UPat.var("y")), sigmoid_like),
  (UPat.var("x")*(((UPat.var("x")+1)*(UPat.var("x")+1)*(UPat.var("x")+1)).reciprocal()), lambda x: sigmoid_like(x, (x+1).reciprocal())),
])

# *** uop expander ***

def _expand_arg_to_idx(args:Tuple[Tuple[int, int], ...], rpk:Dict[int, int]) -> int:
  idx, mul = 0, 1
  for axis,m in args[::-1]:
    idx += rpk[axis] * mul
    mul *= m
  return idx

def _choices_from_args(args:Tuple[Tuple[int, int], ...]) -> List[Dict[int, int]]:
  return [dict(x) for x in itertools.product(*[zip(itertools.repeat(axis), range(m)) for axis,m in args])]

@functools.lru_cache(None)
def _swizzle_args(cargs:Tuple[Tuple[int, int], ...], eargs:Tuple[Tuple[int, int], ...], exclude_args:Tuple[int, ...]) -> List[int]:
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
      elif src.dtype.count > 1:
        # put any input dtype > 1 grouped together
        new_srcs.append(UOp(Ops.VECTORIZE,
                            src.dtype.scalar().vec(expand_sz*src.dtype.count), tuple(src.gep(i) for i in range(src.dtype.count))*expand_sz))
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

def no_vectorized_alu(alu):
  if alu.dtype.vcount == 1: return None
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.VECTORIZE, alu.dtype, alus)

def create_gate(root:UOp) -> Optional[UOp]:
  @functools.lru_cache(None)
  def _gate_srcs(u:UOp, gate:UOp) -> UOp:
    if u.op is Ops.BARRIER: return u
    if u.op is Ops.LOAD and u.src[-1].op is Ops.BARRIER:
      return UOp(u.op, u.dtype, u.src[:-1]+(UOp(Ops.IF, dtypes.void, (gate, u.src[-1])),), u.arg)
    return u if (replace_source:=tuple(_gate_srcs(x, gate) for x in u.src)) == u.src else UOp(u.op, u.dtype, replace_source, u.arg)
  idx = root.src[0]
  if idx.op is Ops.CAST: idx = idx.src[0]
  return None if idx.op is not Ops.INDEX or len(idx.src) == 2 or (ret:=_gate_srcs(root, idx.src[2])) is root else ret

expander = PatternMatcher([
  # double expand
  (UPat(Ops.UNROLL, name="outer", src=(UPat(Ops.UNROLL, name="inner"),)),
   lambda outer, inner: UOp(Ops.UNROLL, outer.dtype, (inner.src[0],), inner.arg+outer.arg)),
  # do expansion
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.WMMA, Ops.LOAD, Ops.STORE, Ops.INDEX, Ops.ASSIGN,
         Ops.VECTORIZE, Ops.IF), name="root", custom_early_reject=set([Ops.UNROLL])), do_expand),
  (UPat(Ops.CONTRACT, name="con"), do_contract),
  # vectorize DEFINE_ACC
  (UPat(Ops.VECTORIZE, src=UPat(Ops.DEFINE_ACC, name="acc"), name="v"), lambda acc,v: acc.replace(dtype=v.dtype)),
  # BARRIERs aren't actually expanded
  (UPat(Ops.BARRIER, src=(UPat(Ops.UNROLL, name="ex"),)),
   lambda ex: UOp(Ops.UNROLL, dtypes.void, (UOp(Ops.BARRIER, dtypes.void, ex.src),)*len(ex.src), ex.arg)),
  # empty UNROLL is NOOP
  (UPat(Ops.UNROLL, src=(UPat.var('x'),), arg=()), lambda x: x),
  # UNROLL GEP (needed for WMMA, generalize this) -> vectorized ALU
  (UPat(Ops.UNROLL, name="ex", src=tuple(UPat.var('x').gep(i)+UPat.var('y').gep(i) for i in range(256 if AMX else 8))),
    lambda ex,x,y: UOp(Ops.UNROLL, ex.dtype, tuple((x+y).gep(i) for i in range(256 if AMX else 8)), ex.arg)),
])

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

def delete_redundant_gates(buf:UOp, idx:UOp, val:UOp, store_gate:UOp, cast:Optional[UOp]=None) -> Optional[UOp]:
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

migrate_indexing = PatternMatcher([
  # create gate MUST BE BEFORE expander
  (UPat(Ops.STORE, name="root"), create_gate),
])

def move_mask(x:UOp, buf:UOp, idx:UOp, mask:UOp, cast:Optional[UOp]=None) -> UOp:
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
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat.any(masked_index:=UPat(Ops.INDEX, src=(UPat(name="buf"), UPat(name="idx"), UPat(name="mask"))),
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

  # initial symbolic + migrate indexing (remove this)
  sink = graph_rewrite(sink, sym+migrate_indexing)

  # expand
  sink = graph_rewrite(sink, sym+expander)

  # devectorize + load_store_indexing
  sink = graph_rewrite(sink, sym+(devectorize+float4_folding if opts is not None and opts.supports_float4 else devectorize)+load_store_indexing)

  # final rules for the renderer (without sym)
  sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL>=2)+pm_render+extra_matcher)
  return sink
