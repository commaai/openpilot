from dataclasses import replace
from tinygrad.dtype import dtypes, DType, truncate
from tinygrad.helpers import flatten, DEBUG, EMULATED_DTYPES, Context, SPEC
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, graph_rewrite, ParamArg
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
  zero = UOp.const(dt, 0)
  if len(uops) == 2: a0, a1 = uops
  elif len(uops) == 4: a0, a1, b0, b1 = uops
  match op:
    case Ops.NEG: return l2i(Ops.SUB, dt, zero, zero, *uops)
    case Ops.CAST if dt in (dtypes.long, dtypes.ulong) and uops[0].dtype not in dtypes.floats:
      return uops[0].cast(l2i_dt[dt]), (uops[0] < 0).where(UOp.const(l2i_dt[dt], -1), UOp.const(l2i_dt[dt], 0))
    case Ops.CAST if dt in (dtypes.long, dtypes.ulong):
      return (lo:=uops[0].cast(l2i_dt[dt])), (uops[0] / 2**32).cast(l2i_dt[dt]) - ((uops[0] < 0) & lo.ne(0)).cast(l2i_dt[dt])
    case Ops.CAST if dt in dtypes.floats:
      small = (a1.eq(0) & (a0 >= 0)) | (a1.eq(-1) & (a0 < 0))
      return small.where(a0.cast(dt), ((a1.cast(dtypes.float32) * (2**32)) + a0.bitcast(dtypes.uint).cast(dtypes.float32)).cast(dt))
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
    case Ops.ADD: return (low:=a0+b0), (a1 + b1).replace(dtype=dt) + (low.bitcast(dtypes.uint) < a0.bitcast(dtypes.uint)).cast(dt)
    case Ops.SUB: return a0 - b0, a1 - b1 - (a0.bitcast(dtypes.uint) < b0.bitcast(dtypes.uint)).cast(dt)
    case Ops.MUL:
      (a00, a01), (b00, b01) = unpack32(a0), unpack32(b0)
      mid = l2i(Ops.ADD, dt, shl(a00*b01, 16).bitcast(dt), shr(a00*b01, 16).bitcast(dt), shl(a01*b00, 16).bitcast(dt), shr(a01*b00, 16).bitcast(dt))
      return l2i(Ops.ADD, dt, *mid, (a00*b00).bitcast(dt), (a01*b01).bitcast(dt) + a0*b1 + a1*b0)
    case Ops.CDIV | Ops.CMOD:
      # TAOCP Algorithm 4.3.1D could be faster here, but must be parameterized over the width of b
      if dt == dtypes.int:
        ua0, ua1, ub0, ub1 = a0.bitcast(dtypes.uint), a1.bitcast(dtypes.uint), b0.bitcast(dtypes.uint), b1.bitcast(dtypes.uint)
        a0, a1 = (a_neg:=a1 < zero).where((n:=l2i(Ops.NEG, dtypes.uint, ua0, ua1))[0], ua0), a_neg.where(n[1], ua1)
        b0, b1 = (b_neg:=b1 < zero).where((n:=l2i(Ops.NEG, dtypes.uint, ub0, ub1))[0], ub0), b_neg.where(n[1], ub1)
      q, r = (z:=UOp.const(dtypes.uint, 0), z), (z, z)
      for i in range(63, -1, -1):
        r = l2i(Ops.SHL, dtypes.uint, *r, UOp.const(dtypes.uint, 1), z)
        r = (r[0] | l2i(Ops.SHR, dtypes.uint, a0, a1, UOp.const(dtypes.uint, i), z)[0] & 1), r[1]
        cond = l2i(Ops.CMPLT, dtypes.uint, *r, b0, b1).logical_not()
        diff = l2i(Ops.SUB, dtypes.uint, *r, b0, b1)
        q = ((q[0] | shl(cond.cast(dtypes.uint), i % 32), q[1]) if i < 32 else (q[0], q[1] | shl(cond.cast(dtypes.uint), i % 32)))
        r = l2i(Ops.WHERE, dtypes.uint, cond, *diff, *r)
      if dt == dtypes.int:
        (nq0, nq1), (nr0, nr1) = l2i(Ops.BITCAST, dt, *l2i(Ops.NEG, dtypes.uint, *q)), l2i(Ops.BITCAST, dt, *l2i(Ops.NEG, dtypes.uint, *r))
        (q0, q1), (r0, r1) = l2i(Ops.BITCAST, dt, *q), l2i(Ops.BITCAST, dt, *r)
        return (a_neg.where(nr0, r0), a_neg.where(nr1, r1)) if op == Ops.CMOD else ((a_neg^b_neg).where(nq0, q0), (a_neg^b_neg).where(nq1, q1))
      return (r[0].bitcast(dt), r[1].bitcast(dt)) if op == Ops.CMOD else (q[0].bitcast(dt), q[1].bitcast(dt))
    case Ops.CMPLT: return (a1 < b1) | ((a1.eq(b1)) & (a0.bitcast(dtypes.uint) < b0.bitcast(dtypes.uint)))
    case Ops.CMPEQ: return a0.eq(b0) & a1.eq(b1)
    case Ops.CMPNE: return a0.ne(b0) | a1.ne(b1)
    case Ops.XOR | Ops.OR | Ops.AND: return UOp(op, dt, src=(a0, b0)), UOp(op, dt, src=(a1, b1))
    case Ops.WHERE: return uops[0].where(uops[1], uops[3]), uops[0].where(uops[2], uops[4])
    case Ops.MAX: return l2i(Ops.WHERE, dt, l2i(Ops.CMPLT, dt, *uops), b0, b1, a0, a1)
    case _: raise NotImplementedError(f"long decomposition of {op} unsupported")

# ***** floats *****
f2f_dt = { f:getattr(dtypes, f"uint{f.bitsize}") for f in dtypes.floats }

def rne(v: UOp, s) -> UOp: return shr(v, s) + ((shr(v, s - 1) & 1) & ((v & ((1 << (s - 1)) - 1)).ne(0).cast(v.dtype) | (shr(v, s) & 1)))

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
    v = f2f_clamp(v.bitcast(fr), to, sat).bitcast(f2f_dt[fr])
    sign, nosign = shr(v, fs - ts) & shl(1, ts - 1), v & (shl(1, fs - 1) - 1)
    norm = (rne(nosign, fm - tm) - shl(fb - tb, tm)).cast(f2f_dt[to])
    underflow = (shr(v, fm) & (shl(1, fe) - 1)) < (1 + fb - tb)
    nan_mantissa = (shl(1, tm) - 1) if to == dtypes.fp8e4m3 else (shr(nosign, fm - tm) & (shl(1, tm) - 1))
    nan = (sign | nan_mantissa | shl(shl(1, te) - 1, tm)).cast(f2f_dt[to])
    is_nan = (shr(v, fm) & (shl(1, fe) - 1)).eq(shl(1, fe) - 1)
    if to in dtypes.fp8_fnuz: return is_nan.where(shl(1, ts - 1), underflow.where(0, sign.cast(f2f_dt[to]) | norm))
    return is_nan.where(nan, sign.cast(f2f_dt[to]) | underflow.where(0, norm))
  else: raise NotImplementedError(f"unsupported decomp {fr} -> {to}")

def f2f_clamp(val:UOp, dt:DType, sat=True) -> UOp:
  e, m = dtypes.finfo(dt)
  if dt in dtypes.fp8_fnuz: max_exp, max_man = (1 << e) - 1, (1 << m) - 1
  else: max_exp, max_man = ((1 << e) - 1, (1 << m) - 2) if dt == dtypes.fp8e4m3 else ((1 << e) - 2, (1 << m) - 1)
  mx = val.const_like(2.0**(max_exp - exponent_bias(dt)) * (1.0 + max_man / (1 << m)))
  sat = mx if dt in dtypes.fp8s and sat else val.const_like(float('inf'))
  # FIXME: CMPLT of nan is undefined
  return val.ne(val).where(val, (val < -mx).where(-sat, (mx < val).where(sat, val)))

def f2f_load(x: UOp, fr:DType, to:DType) -> UOp:
  if (n:=x.max_numel()) == 1: return f2f(x.replace(dtype=f2f_dt[fr]), fr, to)
  return UOp(Ops.STACK, src=tuple(f2f(x.replace(dtype=f2f_dt[fr], src=(reindex(x.src[0], i, 1),)), fr, to) for i in range(n)))

def f2f_store(st, idx, val, fr:DType, to:DType):
  if (n:=val.max_numel()) == 1: return st.replace(src=(idx, f2f(val.bitcast(f2f_dt[to]), to, fr)))
  return UOp.group(*(st.replace(src=(reindex(idx, i, 1), f2f(val.index(i).bitcast(f2f_dt[to]), to, fr))) for i in range(n)))

pm_long_decomp = PatternMatcher([
  (UPat(GroupOp.Defines, src=(UPat.var("sz"),), name="x"), lambda x,sz:
   x.replace(dtype=l2i_dt[x.dtype], arg=replace(x.arg, dtype=l2i_dt[x.dtype]), src=(sz*2,)) if x.dtype in l2i_dt else None),
  (UPat(Ops.INDEX, tuple(l2i_dt.keys()), name='x'), lambda x: reindex(x, x.tag).replace(dtype=l2i_dt[x.dtype]) if x.tag is not None else None),
  (UPat(Ops.STORE, src=(UPat.var('idx'), UPat.var('val', tuple(l2i_dt.keys()))), name='st'), lambda st,idx,val:
   st.replace(src=(idx.rtag(0), val.rtag(0))).group(st.replace(src=(idx.rtag(1), val.rtag(1)))) if val.tag is None else None),
  (UPat(GroupOp.Comparison, src=(UPat.var('a', tuple(l2i_dt.keys())), UPat.var('b', tuple(l2i_dt.keys()))), name="x"), lambda a,b,x:
   l2i(x.op, dt:=l2i_dt[a.dtype], a.rtag(0).cast(dt), a.rtag(1).cast(dt), b.rtag(0).cast(dt), b.rtag(1).cast(dt))),
  (UPat(Ops.CAST, tuple(l2i_dt.keys()), src=(UPat.var('a'),), name="x"), lambda a,x:
   l2i(x.op, x.dtype, a)[x.tag] if x.tag is not None and a.dtype not in l2i_dt else None),
  (UPat(Ops.CAST, tuple(l2i_dt.keys()), src=(UPat.var('a', tuple(l2i_dt.keys())),), name="x"), lambda a,x:
   (a.rtag(0).cast(dt:=l2i_dt[a.dtype]).bitcast(xdt:=l2i_dt[x.dtype]), a.rtag(1).cast(dt).bitcast(xdt))[x.tag]),
  (UPat(Ops.CAST, src=(UPat.var('a', tuple(l2i_dt.keys())),), name="x"), lambda a,x:
   l2i(x.op, x.dtype, a.rtag(0).cast(dt:=l2i_dt[a.dtype]), a.rtag(1).cast(dt)) if x.dtype not in l2i_dt and a.tag is None else None),
  (UPat((*(GroupOp.ALU - GroupOp.Comparison), Ops.BITCAST), tuple(l2i_dt.keys()), name="x"), lambda x:
   l2i(x.op, l2i_dt[x.dtype], *flatten((a.rtag(0).cast(dt:=l2i_dt[x.src[-1].dtype]), a.rtag(1).cast(dt))
                                       if a.dtype in l2i_dt else (a,) for a in x.src))[x.tag] if x.tag is not None else None),
  (UPat(Ops.LOAD, tuple(l2i_dt.keys()), src=(UPat.var('idx'),), name='x'), lambda x,idx:
   x.replace(dtype=l2i_dt[x.dtype], src=(reindex(idx, x.tag).replace(dtype=l2i_dt[x.dtype]),))),
  (UPat(Ops.CONST, tuple(l2i_dt.keys()), name='x'), lambda x:
   UOp.const(dt:=l2i_dt[x.dtype], truncate[dt]((x.arg >> 32) if x.tag == 1 else (x.arg & 0xFFFFFFFF))))
])

# float decomposition patterns - ctx is (fr, to) tuple
pm_float_decomp = PatternMatcher([
  (UPat((*GroupOp.Defines, Ops.INDEX, Ops.SHRINK), name="x"), lambda ctx,x:
   x.replace(dtype=f2f_dt[ctx[0]], arg=replace(x.arg, dtype=f2f_dt[ctx[0]]) if isinstance(x.arg, ParamArg) else x.arg, tag=ctx[0])
   if x.dtype == ctx[0] and (x.op is not Ops.INDEX or x.src[0].op not in {Ops.LOAD, Ops.STACK}) else None),
  (UPat(Ops.LOAD, dtypes.floats, name="x"), lambda ctx,x: f2f_load(x, *ctx) if x.dtype == ctx[0] else None),
  # bitcasted load should just replace load
  (UPat(Ops.BITCAST, src=(UPat(Ops.LOAD, name="ld"),), name="bc"), lambda ctx,bc,ld:
   ld.replace(dtype=f2f_dt[ctx[0]]).bitcast(bc.dtype) if ld.dtype == ctx[0] else None),
  # bitcast from
  (UPat(Ops.BITCAST, src=(UPat.var("x", dtypes.floats),), name="bc"), lambda ctx,bc,x:
   bc.replace(src=(f2f(x.bitcast(f2f_dt[ctx[1]]), ctx[1], ctx[0]),)) if x.dtype == ctx[1] and bc.dtype.bitsize == ctx[0].bitsize else None),
  # bitcast to
  (UPat(Ops.BITCAST, src=(UPat.var("x"),), name="bc"), lambda ctx,bc,x:
   f2f(x.bitcast(f2f_dt[ctx[0]]), ctx[0], ctx[1]) if bc.dtype == ctx[0] else None),
  (UPat(Ops.CAST, dtypes.floats, src=(UPat.var("val"),), name="x"), lambda ctx,x,val:
   f2f_clamp(val.cast(ctx[1]), ctx[0]) if x.dtype == ctx[0] else None),
  (UPat(GroupOp.All-{Ops.BITCAST}, dtypes.floats, name="x"), lambda ctx,x:
   x.replace(dtype=ctx[1], src=tuple(s.cast(ctx[1]) if s.dtype == ctx[0] else s for s in x.src))
   if x.dtype == ctx[0] else None),
  (UPat(Ops.STORE, src=(UPat.var("idx"), UPat(Ops.BITCAST, dtypes.floats, name="val")), name='st'), lambda ctx,st,idx,val:
   st.replace(src=(idx, val.replace(dtype=f2f_dt[ctx[0]]))) if val.dtype == ctx[0] and idx.tag == ctx[0] else None),
  (UPat(Ops.STORE, src=(UPat.var("idx"), UPat.var("val", dtypes.floats)), name='st'), lambda ctx,st,idx,val:
   f2f_store(st, idx, val, *ctx) if val.dtype == ctx[1] and (idx:=idx.src[0] if idx.op == Ops.CAST else idx).tag == ctx[0] else None),
])

def do_dtype_decomps(sink:UOp, ctx:tuple[set[DType], Renderer]) -> UOp:
  def _should_emulate(dt): return dt in EMULATED_DTYPES.tolist(dtypes) or dt not in ctx[1].supported_dtypes()
  # NOTE: dtype decomp creates intermediate UOps that don't follow the spec (e.g. half LOAD on ushort BUFFER)
  with Context(SPEC=min(SPEC.value, 1)):
    for fr in sorted(filter(_should_emulate, ctx[0])):
      to = dtypes.int if fr == dtypes.long else dtypes.half if not _should_emulate(dtypes.half) and fr in dtypes.fp8s else dtypes.float
      if DEBUG >= 2: print(f"emulating {fr} as {to}")
      pm = pm_float_decomp if fr in dtypes.floats else pm_long_decomp
      sink = graph_rewrite(sink, pm, name=f"decomp {fr} -> {to}", ctx=(fr, to), bottom_up=True)
  ctx[0].clear()
  return sink

pm_dtype_decomps = PatternMatcher([
  # detect dtypes to decompose
  (UPat(GroupOp.All, (*dtypes.fp8s, dtypes.bfloat16, dtypes.half, dtypes.long, dtypes.ulong), name="x"), lambda x,ctx:
   ctx[0].add({dtypes.ulong:dtypes.long}.get(dt:=x.dtype, dt))),
  # do the rewrites
  (UPat(Ops.SINK, name="sink"), do_dtype_decomps),
])
