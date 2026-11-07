from dataclasses import dataclass, field, replace
from tinygrad.dtype import dtypes, DType, AddrSpace, truncate
from tinygrad.helpers import flatten, DEBUG, EMULATED_DTYPES
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, graph_rewrite, shape_to_shape_arg
from tinygrad.uop.weak import commit_weak_consts
from tinygrad.renderer import Renderer
from tinygrad.codegen.decomp.transcendental import exponent_bias, shl, shr

# ***** long as 2 ints *****

l2i_dt = {dtypes.long: dtypes.int, dtypes.ulong: dtypes.uint}
def unpack32(v:UOp) -> tuple[UOp, UOp]: return v.bitcast(dtypes.uint) & 0xFFFF, shr(v.bitcast(dtypes.uint), 16)
def reindex(idx:UOp, off:int, mul=2) -> UOp:
  if idx.op is Ops.SHRINK:
    assert mul == 1, "can't reindex SHRINK with mul != 1"
    return idx.replace(op=Ops.INDEX, src=(idx.src[0], idx.src[1]+off))
  return idx.replace(src=(idx.src[0], idx.src[1]*mul+off, *idx.src[2:]))

# 4.3.1 is the relevant section in TAOCP
def l2i(op: Ops, dt: DType, *uops:UOp):
  zero = UOp.const(0, dt)
  if len(uops) == 2: a0, a1 = uops
  elif len(uops) == 4: a0, a1, b0, b1 = uops
  match op:
    case Ops.NEG: return l2i(Ops.SUB, dt, zero, zero, *uops)
    case Ops.CAST if dt in (dtypes.long, dtypes.ulong) and uops[0].dtype not in dtypes.floats:
      # the high word is the sign extension, and unsigned and bool sources zero extend
      x, lo = uops[0], uops[0].cast(l2i_dt[dt])
      if x.dtype is dtypes.bool or x.dtype in dtypes.uints: return lo, lo.const_like(0)
      return lo, (x < x.const_like(0)).where(lo.const_like(-1), lo.const_like(0))
    case Ops.CAST if dt in (dtypes.long, dtypes.ulong):
      return (lo:=uops[0].cast(l2i_dt[dt])), (uops[0] / 2**32).cast(l2i_dt[dt]) - ((uops[0] < 0) & lo.ne(0))
    case Ops.CAST if dt in dtypes.floats:
      small = (a1.eq(0) & (a0 >= 0)) | (a1.eq(-1) & (a0 < 0))
      cdt = dt if dt == dtypes.float64 else dtypes.float32
      return small.where(a0.cast(dt), ((a1.cast(cdt) * (2**32)) + a0.bitcast(dtypes.uint).cast(cdt)).cast(dt))
    case Ops.CAST: return a0.bitcast(dtypes.uint).cast(dt)
    case Ops.BITCAST: return a0.bitcast(dt), a1.bitcast(dt)
    case Ops.SHL:
      a0u, a1u, n = a0.bitcast(dtypes.uint), a1.bitcast(dtypes.uint), (b0 & 31).cast(dtypes.uint)
      lo, hi = (a0u << n).bitcast(dt), ((a1u << n) | ((a0u >> 1) >> (31 - n))).bitcast(dt)
      return (b0 >= 32).where(zero, lo), (b0 >= 32).where(lo, hi)
    case Ops.SHR:
      a0u, a1u, n = a0.bitcast(dtypes.uint), a1.bitcast(dtypes.uint), (b0 & 31).cast(dtypes.uint)
      lo, hi = ((a0u >> n) | ((a1u << 1) << (31 - n))).bitcast(dt), a1 >> (b0 & 31)
      fill = a1 >> 31 if dt == dtypes.int else zero  # vacated high word: sign bits when signed, else 0
      return (b0 >= 32).where(hi, lo), (b0 >= 32).where(fill, hi)
    case Ops.ADD: return (low:=a0+b0), a1 + b1 + (low.bitcast(dtypes.uint) < a0.bitcast(dtypes.uint))
    case Ops.SUB: return a0 - b0, a1 - b1 - (a0.bitcast(dtypes.uint) < b0.bitcast(dtypes.uint))
    case Ops.MUL:
      (a00, a01), (b00, b01) = unpack32(a0), unpack32(b0)
      mid = l2i(Ops.ADD, dt, shl(a00*b01, 16).bitcast(dt), shr(a00*b01, 16).bitcast(dt), shl(a01*b00, 16).bitcast(dt), shr(a01*b00, 16).bitcast(dt))
      return l2i(Ops.ADD, dt, *mid, (a00*b00).bitcast(dt), (a01*b01).bitcast(dt) + a0*b1 + a1*b0)
    case Ops.TRUNCDIV | Ops.TRUNCMOD:
      # TAOCP Algorithm 4.3.1D could be faster here, but must be parameterized over the width of b
      if dt == dtypes.int:
        ua0, ua1, ub0, ub1 = a0.bitcast(dtypes.uint), a1.bitcast(dtypes.uint), b0.bitcast(dtypes.uint), b1.bitcast(dtypes.uint)
        a0, a1 = (a_neg:=a1 < zero).where((n:=l2i(Ops.NEG, dtypes.uint, ua0, ua1))[0], ua0), a_neg.where(n[1], ua1)
        b0, b1 = (b_neg:=b1 < zero).where((n:=l2i(Ops.NEG, dtypes.uint, ub0, ub1))[0], ub0), b_neg.where(n[1], ub1)
      q, r = (z:=UOp.const(0, dtypes.uint), z), (z, z)
      for i in range(63, -1, -1):
        r = l2i(Ops.SHL, dtypes.uint, *r, UOp.const(1, dtypes.uint), z)
        r = (r[0] | l2i(Ops.SHR, dtypes.uint, a0, a1, UOp.const(i, dtypes.uint), z)[0] & 1), r[1]
        cond = l2i(Ops.CMPLT, dtypes.uint, *r, b0, b1).logical_not()
        diff = l2i(Ops.SUB, dtypes.uint, *r, b0, b1)
        q = ((q[0] | shl(cond.cast(dtypes.uint), i % 32), q[1]) if i < 32 else (q[0], q[1] | shl(cond.cast(dtypes.uint), i % 32)))
        r = l2i(Ops.WHERE, dtypes.uint, cond, *diff, *r)
      if dt == dtypes.int:
        (nq0, nq1), (nr0, nr1) = l2i(Ops.BITCAST, dt, *l2i(Ops.NEG, dtypes.uint, *q)), l2i(Ops.BITCAST, dt, *l2i(Ops.NEG, dtypes.uint, *r))
        (q0, q1), (r0, r1) = l2i(Ops.BITCAST, dt, *q), l2i(Ops.BITCAST, dt, *r)
        return (a_neg.where(nr0, r0), a_neg.where(nr1, r1)) if op == Ops.TRUNCMOD else ((a_neg^b_neg).where(nq0, q0), (a_neg^b_neg).where(nq1, q1))
      return r if op == Ops.TRUNCMOD else q
    case Ops.CMPLT: return (a1 < b1) | ((a1.eq(b1)) & (a0.bitcast(dtypes.uint) < b0.bitcast(dtypes.uint)))
    case Ops.CMPEQ: return a0.eq(b0) & a1.eq(b1)
    case Ops.CMPNE: return a0.ne(b0) | a1.ne(b1)
    case Ops.XOR | Ops.OR | Ops.AND: return UOp(op, src=(a0, b0)), UOp(op, src=(a1, b1))
    case Ops.WHERE: return uops[0].where(uops[1], uops[3]), uops[0].where(uops[2], uops[4])
    case Ops.MAX: return l2i(Ops.WHERE, dt, l2i(Ops.CMPLT, dt, *uops), b0, b1, a0, a1)
    case _: raise NotImplementedError(f"long decomposition of {op} unsupported")

def l2i_define(x:UOp) -> UOp:
  # cannot decomp a Variable
  if x.addrspace == AddrSpace.ALU: raise RuntimeError(f"long decomposition of variable {x.arg.name} unsupported")
  size = shape_to_shape_arg((x.max_numel()*2,)) if x.shape else x.src[0]
  return x.replace(arg=replace(x.arg, dtype=l2i_dt[x.dtype]), src=(size, *x.src[1:]))

# the pair of a long is the long bitcast to two words, its words are the pair's lanes
def l2i_pair(x:UPat) -> UPat: return UPat(Ops.BITCAST, src=(UPat(Ops.STACK, src=(x,)),), name='p')
def l2i_words(*uops:UOp) -> list[UOp]:
  return flatten(((p:=UOp(Ops.STACK, src=(u,)).bitcast(l2i_dt[u.dtype])).index(0), p.index(1)) if u.dtype in l2i_dt else (u,) for u in uops)

# ***** floats *****
f2f_dt = { f:getattr(dtypes, f"uint{f.bitsize}") for f in dtypes.floats }

def rne(v: UOp, s) -> UOp: return shr(v, s) + ((shr(v, s - 1) & 1) & ((v & ((1 << (s - 1)) - 1)).ne(0) | (shr(v, s) & 1)))

def f2f(v, fr:DType, to:DType, sat=True):
  fs, fb, (fe, fm), ts, tb, (te, tm) = fr.bitsize, exponent_bias(fr), dtypes.finfo(fr), to.bitsize, exponent_bias(to), dtypes.finfo(to)
  # NB: denormals are zero!
  if fe <= te and fm < tm:
    sign, nosign = shl((v & shl(1, fs-1)).cast(f2f_dt[to]), ts - fs), (v & (shl(1, fs-1) - 1)).cast(f2f_dt[to])
    exp, norm = shr(nosign, fm), shl(nosign, tm - fm) + shl(tb - fb, tm)
    nan = shl(nosign, tm - fm) | shl((shl(1, te) - 1), tm)
    if fr in dtypes.fp8_fnuz:
      fnuz_nan = sign.ne(0) & nosign.eq(0)
      qnan = shl(shl(1, te) - 1, tm) | shl(1, tm - 1)
      return fnuz_nan.where(qnan, sign | exp.eq(0).where(0, norm)).bitcast(to)
    # fp8e4m3 has only one nan
    is_nan = (nosign.eq(shl(1, fm + fe) - 1) if fr == dtypes.fp8e4m3 else exp.eq(shl(1, fe) - 1))
    return (sign | exp.eq(0).where(0, is_nan.where(nan, norm))).bitcast(to)
  elif fe >= te and fm > tm:
    sign, nosign = shr(v, fs - ts) & shl(1, ts - 1), v & (shl(1, fs - 1) - 1)
    # denormals are zero, but within half a denormal ulp of the smallest normal rounds up to it
    min_norm = shl(1 + fb - tb, fm)
    underflow = nosign < max(min_norm - shl(1, fm - tm), shl(1, fm))
    norm = rne(nosign.maximum(min_norm), fm - tm) - shl(fb - tb, tm)
    # top is one past the largest finite, inf or the nan of formats without inf
    top = 0x7f if to == dtypes.fp8e4m3 else 0x80 if to in dtypes.fp8_fnuz else shl(shl(1, te) - 1, tm)
    norm = norm.minimum(top - (sat and to in dtypes.fp8s))
    inf, nan = shl(shl(1, fe) - 1, fm), top | (shl(1, tm) - 1 if to in dtypes.fp8s else shl(1, tm - 1))
    # fnuz has one nan and no -0
    if to in dtypes.fp8_fnuz: return (nosign >= inf).where(top, underflow.where(0, sign | norm)).cast(f2f_dt[to])
    return (sign | (nosign > inf).where(nan, nosign.eq(inf).where(top, underflow.where(0, norm)))).cast(f2f_dt[to])
  else: raise NotImplementedError(f"unsupported decomp {fr} -> {to}")

@dataclass(frozen=True)
class FloatDecompCtx:
  fr: DType
  to: DType

def f2f_load(x: UOp, fr:DType, to:DType) -> UOp:
  storage_idx = graph_rewrite(x.src[0], pm_float_decomp, ctx=FloatDecompCtx(fr, to), bottom_up=True)
  if (n:=x.max_numel()) == 1: return f2f(storage_idx.load(*x.src[1:]), fr, to)
  return UOp(Ops.STACK, src=tuple(f2f(reindex(storage_idx, i, 1).load(*x.src[1:]), fr, to) for i in range(n)))

def f2f_store(st, idx, val, fr:DType, to:DType):
  if (n:=val.max_numel()) == 1: return st.replace(src=(idx, f2f(val.bitcast(f2f_dt[to]), to, fr)))
  return UOp.group(*(st.replace(src=(reindex(idx, i, 1), f2f(val.index(i).bitcast(f2f_dt[to]), to, fr))) for i in range(n)))

def f2f_rewrite(ctx:FloatDecompCtx, x:UOp) -> UOp: return graph_rewrite(x, pm_float_decomp, ctx=ctx, bottom_up=True)

pm_long_decomp: PatternMatcher = PatternMatcher([
  # the decomp's own bottom-up rewrite can mint bare consts mid-flight: word splitting commits them at the long sibling's dtype
  (UPat(GroupOp.All, name='x'), lambda x: commit_weak_consts(x, next((s.dtype for s in x.src if s.dtype in l2i_dt), None))),
  (UPat(GroupOp.Defines, tuple(l2i_dt.keys()), name="x"), l2i_define),
  # long stores, comparisons and casts out of long consume the words of their long srcs, a stored word is no longer long
  (UPat(Ops.STORE, src=(UPat.var('idx', tuple(l2i_dt.keys())), UPat.var('val', tuple(l2i_dt.keys()))), name='st'), lambda st,idx,val:
   UOp.group(*(st.replace(src=(reindex(idx, w), v)) for w,v in enumerate(l2i_words(val))))),
  (UPat(GroupOp.Comparison, src=[UPat.var('a', tuple(l2i_dt.keys())), UPat()], name="x"), lambda a,x: l2i(x.op, l2i_dt[a.dtype], *l2i_words(*x.src))),
  (UPat(Ops.CAST, src=(UPat.var('a', tuple(l2i_dt.keys())),), name="x"), lambda a,x:
   l2i(x.op, x.dtype, *l2i_words(a)) if x.dtype not in l2i_dt else None),
  # a pair becomes the STACK of the long's two words
  (l2i_pair(UPat(Ops.LOAD, src=(UPat.var('idx'),))), lambda p,idx: UOp(Ops.STACK, src=tuple(reindex(idx, w).load() for w in (0, 1)))),
  # a const splits by value, the general CAST below would drop its high word
  (l2i_pair(UPat(Ops.CAST, src=(UPat(Ops.CONST, name='c'),))), lambda p,c:
   UOp(Ops.STACK, src=tuple(UOp.const(truncate[p.dtype](c.val >> (32*w)), p.dtype) for w in (0, 1)))),
  (l2i_pair(UPat(Ops.CAST, src=(UPat.var('a'),), name='x')), lambda p,a,x:
   UOp(Ops.STACK, src=l2i(x.op, x.dtype, a)) if a.dtype not in l2i_dt else None),
  # a cast between long dtypes reinterprets the words, a bare const src commits at the long dtype
  (l2i_pair(UPat(GroupOp.Elementwise, name='x')), lambda p,x:
   UOp(Ops.STACK, src=l2i(Ops.BITCAST if x.op is Ops.CAST else x.op, p.dtype, *l2i_words(*commit_weak_consts(x, x.dtype).src)))),
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, name='s'), UPat.cvar('w'))), lambda s,w: s.src[w.arg]),
])

# float decomposition patterns
pm_float_decomp: PatternMatcher = PatternMatcher([
  (UPat(GroupOp.Defines, name="x"), lambda ctx,x:
   x.replace(arg=replace(x.arg, dtype=f2f_dt[ctx.fr])) if x.dtype == ctx.fr else None),
  # INDEX into a LOAD/STACK selects a lane of an already converted value, the load rules below own those
  (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(GroupOp.All-{Ops.LOAD, Ops.STACK}),), allow_any_len=True, name="x"), lambda ctx,x:
   x.replace(src=(f2f_rewrite(ctx, x.src[0]), *x.src[1:])) if x.dtype == ctx.fr else None),
  (UPat(Ops.LOAD, dtypes.floats, name="x"), lambda ctx,x: f2f_load(x, ctx.fr, ctx.to) if x.dtype == ctx.fr else None),
  # bitcasted load should just replace load
  (UPat(Ops.BITCAST, src=(UPat(Ops.LOAD, name="ld"),), name="bc"), lambda ctx,bc,ld:
   f2f_rewrite(ctx, ld.src[0]).load(*ld.src[1:]).bitcast(bc.dtype) if ld.dtype == ctx.fr else None),
  # bitcast from
  (UPat(Ops.BITCAST, src=(UPat.var("x", dtypes.floats),), name="bc"), lambda ctx,bc,x:
   bc.replace(src=(f2f(x.bitcast(f2f_dt[ctx.to]), ctx.to, ctx.fr),)) if x.dtype == ctx.to and bc.dtype.bitsize == ctx.fr.bitsize else None),
  # bitcast to
  (UPat(Ops.BITCAST, src=(UPat.var("x"),), name="bc"), lambda ctx,bc,x:
   f2f(x.bitcast(f2f_dt[ctx.fr]), ctx.fr, ctx.to) if bc.dtype == ctx.fr else None),
  (UPat(Ops.CAST, dtypes.floats, src=(UPat.var("val"),), name="x"), lambda ctx,x,val:
   f2f(f2f(val.cast(ctx.to).bitcast(f2f_dt[ctx.to]), ctx.to, ctx.fr), ctx.fr, ctx.to) if x.dtype == ctx.fr else None),
  (UPat(GroupOp.ALU|{Ops.STACK, Ops.INDEX}, dtypes.floats, name="x"), lambda ctx,x:
   x.replace(src=tuple(s.cast(ctx.to) if s.dtype == ctx.fr else s for s in x.src)) if x.dtype == ctx.fr else None),
  (UPat(Ops.STORE, src=(UPat.var("idx"), UPat(Ops.BITCAST, dtypes.floats, name="val")), name='st'), lambda ctx,st,idx,val:
   st.replace(src=(idx, val.src[0].bitcast(f2f_dt[ctx.fr]))) if val.dtype == ctx.fr and idx.dtype == f2f_dt[ctx.fr] else None),
  (UPat(Ops.STORE, src=(UPat.var("idx").or_casted(), UPat.var("val", dtypes.floats)), name='st'), lambda ctx,st,idx,val:
   f2f_store(st, idx, val, ctx.fr, ctx.to) if val.dtype == ctx.to and idx.dtype == f2f_dt[ctx.fr] else None),
])

@dataclass
class DTypeDecompCtx:
  renderer: Renderer
  dtypes: set[DType] = field(default_factory=set)

def do_dtype_decomps(sink:UOp, ctx:DTypeDecompCtx) -> UOp:
  def _should_emulate(dt): return dt in EMULATED_DTYPES.tolist(dtypes) or dt not in ctx.renderer.supported_dtypes()
  for fr in sorted(filter(_should_emulate, ctx.dtypes)):
    to = dtypes.int if fr == dtypes.long else dtypes.float
    if DEBUG >= 2: print(f"emulating {fr} as {to}")
    pm = pm_float_decomp if fr in dtypes.floats else pm_long_decomp
    sink = graph_rewrite(sink, pm, name=f"decomp {fr} -> {to}", ctx=FloatDecompCtx(fr, to), bottom_up=True)
  ctx.dtypes.clear()
  return sink

pm_dtype_decomps = PatternMatcher([
  # detect dtypes to decompose
  (UPat(GroupOp.All, (*dtypes.fp8s, dtypes.bfloat16, dtypes.half, dtypes.long, dtypes.ulong), name="x"), lambda x,ctx:
   ctx.dtypes.add({dtypes.ulong:dtypes.long}.get(dt:=x.dtype, dt))),
  # do the rewrites
  (UPat(Ops.SINK, name="sink"), do_dtype_decomps),
])
