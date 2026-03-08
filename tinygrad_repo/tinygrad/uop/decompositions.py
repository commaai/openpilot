from typing import Callable
import math, functools
from tinygrad.dtype import dtypes, DType, promo_lattice, truncate
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import flatten, polyN
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher

TRANSCENDENTAL_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64)

def _lazy_map_numbers(x:UOp, inf:UOp, _inf:UOp, nan:UOp, ratio:UOp):
  """replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio"""
  return x.ne(math.inf).where(x.ne(x).where(nan, x.ne(-math.inf).where(ratio, _inf)), inf)

# *** helper functions for bit manipulation ***
def mantissa_bits(d:DType) -> int: return dtypes.finfo(d.scalar())[1]
def exponent_bias(d:DType) -> int: return {dtypes.float64: 1023, dtypes.float32: 127, dtypes.float16: 15}[d.scalar()]
def exponent_mask(d:DType) -> int: return {dtypes.float64: 2047, dtypes.float32: 255, dtypes.float16: 31}[d.scalar()]

# **** utils ****
def shr(x:UOp, y:int) -> UOp: return x // (2**y)
def shl(x:UOp, y:int) -> UOp: return x * (2**y)

def rintk(d:UOp) -> UOp:
  """round d:float to int away from 0"""
  out_dtype = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype.scalar()].vec(d.dtype.vcount)
  return (d + (d<0.0).where(d.const_like(-0.5), d.const_like(0.5))).cast(out_dtype)

def pow2if(q:UOp, float_dtype:DType):
  """cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]"""
  out_dtype = {dtypes.int64: dtypes.float64, dtypes.int32: dtypes.float32, dtypes.int16: float_dtype.scalar()}[q.dtype.scalar()].vec(q.dtype.vcount)
  return shl(q + exponent_bias(out_dtype), mantissa_bits(out_dtype)).bitcast(out_dtype)

def ilogb2k(d:UOp) -> UOp:
  """calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf)."""
  assert d.dtype.scalar() in TRANSCENDENTAL_DTYPES
  dint = d.bitcast({dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype.scalar()].vec(d.dtype.vcount))
  # -1 <= ilog2bk(d) <= 128
  return (shr(dint, mantissa_bits(d.dtype)) & exponent_mask(d.dtype)) - exponent_bias(d.dtype)

def ldexp3k(d:UOp, e:UOp) -> UOp:
  """d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number."""
  assert d.dtype.scalar() in TRANSCENDENTAL_DTYPES and e.dtype.scalar() in TRANSCENDENTAL_DTYPES
  dtype = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype.scalar()].vec(d.dtype.count)
  m1 = d.bitcast(dtype)
  m2 = shl(e.cast(dtype), mantissa_bits(d.dtype))
  return (m1 + m2).bitcast(d.dtype)

def ldexp2k(d:UOp, e:UOp) -> UOp:
  """d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal."""
  assert d.dtype.scalar() in TRANSCENDENTAL_DTYPES and e.dtype.scalar() in (dtypes.int16, dtypes.int32, dtypes.int64)
  return (d * pow2if(shr(e, 1), d.dtype)) * pow2if(e - shr(e, 1), d.dtype)

def frexp(v:UOp) -> tuple[UOp, UOp]:
  """frexp(v) -> (mantissa, exponent) assuming v != 0"""
  assert v.dtype.scalar() in TRANSCENDENTAL_DTYPES
  # m1 = masks for mantissa, m2 = masks to normalize the mantissa.
  m1 = {dtypes.float64: 0x000FFFFFFFFFFFFF, dtypes.float32: 0x807FFFFF, dtypes.float16: 0x83FF}[v.dtype.scalar()]
  m2 = {dtypes.float64: 0x3FE0000000000000, dtypes.float32: 0x3F000000, dtypes.float16: 0x3800}[v.dtype.scalar()]
  bits = v.bitcast({dtypes.float64: dtypes.uint64, dtypes.float32: dtypes.uint32, dtypes.float16: dtypes.uint16}[v.dtype.scalar()].vec(v.dtype.count))
  exponent = shr(bits, mantissa_bits(v.dtype)) & exponent_mask(v.dtype)
  # Set the exponent bits appropriately to normalize the mantissa into the range of [0.5, 1.0).
  mantissa = ((bits & m1) | m2).bitcast(v.dtype)
  exp = exponent - exponent_bias(v.dtype) + 1
  return mantissa, exp

# *** reduction algorithms for sine ***
def payne_hanek_reduction(d:UOp) -> tuple[UOp, UOp]:
  """
  Performs Payne-Hanek Reduction: computes the remainder of `d` modulo pi/2 for the values `d` where
    39800.0 <= d <= +Inf
  Returns a tuple of `(r, q)`:
  - `r`[d.dtype] is the reminder value corresponding to `round_to_nearest(x % pi/2)`.
  - `q`[int32] is an integer, and q % 4 is corresponding to the quadrant of the original angle `d`.
  """
  assert d.dtype.scalar() in TRANSCENDENTAL_DTYPES
  # https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c/30465751#30465751
  # 190 bits of 2/pi for Payne-Hanek style argument reduction
  two_over_pi_f = [0x00000000, 0x28be60db, 0x9391054a, 0x7f09d5f4, 0x7d4d3770, 0x36d8a566, 0x4f10e410]

  intermediate_dtype = dtypes.float32.vec(d.dtype.count) if d.dtype.base.scalar() == dtypes.float16 else d.dtype

  f, e = frexp(d)
  ia = (f.cast(intermediate_dtype) * 4.294967296e9).cast(dtypes.uint64)
  # extract 96 relevant bits of 2/pi based on magnitude of argument
  i = shr(e.cast(dtypes.uint64), 5)
  e = e.cast(dtypes.int32) & 31
  offset = 32 - e

  def _take(an:UOp, offset:int, count:int=0) -> UOp:
    """an = two_over_pi_f[i+offset]"""
    if count+offset < len(two_over_pi_f) - 1:
      an = i.ne(count).where(_take(an, offset, count=count+1), an.const_like(two_over_pi_f[count+offset]))
    return an
  def _shl_lazy(x:UOp, y:UOp): return (x.cast(dtypes.uint64) * pow2if(y, d.dtype).cast(dtypes.uint64)).cast(dtypes.uint32)
  def _shr_lazy(x:UOp, y:UOp): return (x.cast(dtypes.uint64) // pow2if(y, d.dtype).cast(dtypes.uint64)).cast(dtypes.uint32)

  a = [_take(UOp.const(dtypes.uint32.vec(d.dtype.count), 0), i) for i in range(4)]
  #  (two_over_pi_f[Int(i) + n] << e) | (two_over_pi_f[Int(i) + n+1] >> (nbits - e))
  # Note: e >= 1 for all numbers d >= 1.0. assume e != 0
  hi = _shl_lazy(a[0], e) | _shr_lazy(a[1], offset)
  mi = _shl_lazy(a[1], e) | _shr_lazy(a[2], offset)
  lo = _shl_lazy(a[2], e) | _shr_lazy(a[3], offset)

  def _hp_mul(x:UOp, y:UOp) -> UOp: return x.cast(dtypes.uint64) * y.cast(dtypes.uint64)
  # compute x * 2/pi
  p = shl(_hp_mul(ia, hi), 32) + _hp_mul(ia, mi) + shr(_hp_mul(ia, lo), 32)

  # round quotient to nearest
  q = shr(p, 62).cast(dtypes.int32)
  p = p & 0x3fffffffffffffff
  r = (p.cast(intermediate_dtype) * (3.4061215800865545e-19)).cast(d.dtype)

  # if fraction >= 0.5, r -= pi/2, q += 1
  return (f<0.5).where(r, r - math.pi/2), (f<0.5).where(q, q + 1)

def cody_waite_reduction(d:UOp) -> tuple[UOp, UOp]:
  """
  Performs Cody-Waite Reduction: computes the reminder of `d` modulo pi/2 for the values `d` where
      0 <= abs(d) <= 39800.0
  Returns a tuple of `(r, q)`, where the output format is the same as that of `payne_hanek_reduction`.
  """
  def _reduce_d(x:UOp, q:UOp):
    # https://github.com/shibatch/sleef/blob/4e08851f59fc2b545f9c393c6a23dfd311a26308/src/libm/sleefdp.c#L789-L823
    if x.dtype.scalar() == dtypes.float64:
      # https://github.com/shibatch/sleef/blob/f6d8a841fbfddd26ce712834d4da220cd76048fb/src/common/misc.h#L77
      PI_A, PI_B, PI_C, PI_D = 3.1415926218032836914, 3.1786509424591713469e-08, 1.2246467864107188502e-16, 1.2736634327021899816e-24
      d = qdh * -PI_A + x
      d = q * -PI_A + d
      d = qdh * -PI_B + d
      d = q * -PI_B + d
      d = qdh * -PI_C + d
      d = q * -PI_C + d
      d = (qdh + q) * -PI_D + d
    elif x.dtype.scalar() == dtypes.float16:
      # [FIXME] when reducing `d`, FP16 needs FP32 precision to achieve 1.0 ULP precision.
      d = _reduce_d(x.cast(dtypes.float32), q.cast(dtypes.float32)).cast(dtypes.float16)
    else:
      # https://github.com/shibatch/sleef/blob/4e08851f59fc2b545f9c393c6a23dfd311a26308/src/libm/sleefsp.c#L464-L503
      d = q * -3.1414794921875 + x
      d = q * -0.00011315941810607910156 + d
      d = q * -1.9841872589410058936e-09 + d
      d = q * -1.2154201256553420762e-10 + d
    return d

  m_1_pi = 0.318309886183790671537767526745028724
  qdh = (d * (m_1_pi / 2.0**24)).cast(dtypes.int64).cast(d.dtype) * (2.0**24)
  quadrant = rintk(d * m_1_pi -qdh) if d.dtype.base.scalar() == dtypes.float64 else rintk(d * m_1_pi)
  return _reduce_d(d, quadrant.cast(d.dtype)), quadrant.cast(dtypes.int32)

# *** approximate sine on small angle. ***
def trig_poly(d:UOp, coeff32, coeff64): return d * (polyN(d*d, coeff64) if d.dtype.scalar() == dtypes.float64 else polyN(d*d, coeff32))
# approximate sine on [-pi/2, pi/2]
def sin_poly(d:UOp) -> UOp:
  return trig_poly(d, [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938, 1.0],
                      [-7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10,
                       -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815,
                       -0.166666666666666657414808,    1.0])

def _ifand(q:UOp, n:int): return (q & n).ne(0)

def sin_poly_small(d:UOp, q:UOp) -> UOp:
  r = sin_poly(d)
  return r * _ifand(q, 1).where(r.const_like(-1), r.const_like(1))

def sin_poly_large(d:UOp, q:UOp) -> UOp:
  r = sin_poly(d + _ifand(q, 1).where(d.const_like(math.pi / 2), d.const_like(0)))
  return r * _ifand(q, 2).where(r.const_like(-1), r.const_like(1))

# *** toplevel functions for xsin/xlog2/xexp2 ***

def xsin(d:UOp, fast:bool=False, switch_over:float=30.0) -> UOp:
  """
  Implements a 1.0 ULP approximation for Ops.SIN.
  - fast=True assumes x <= switch_over.
  - switch_over is the threshold for switching to payne_hanek_reduction.
  """
  assert d.dtype.scalar() in TRANSCENDENTAL_DTYPES
  # mask +-inf/nan as zero
  x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  # x_sign = sign(x)
  x_sign = x.ne(0).where((x<0).where(x.const_like(-1), x.const_like(1)), x.const_like(0))
  x_abs = x * x_sign
  r, q = (cody_waite_reduction if fast else payne_hanek_reduction)(x_abs)
  if fast: result = sin_poly_small(r, q)
  else:
    # Payne Hanek Reduction assumes abs(x) >= pi/4, so for smaller values, use cody_waite_reduction.
    r_small, q_small = cody_waite_reduction(x_abs)
    result = (x_abs<switch_over).where(sin_poly_small(r_small, q_small), sin_poly_large(r, q))
  # adjusts the sign for abs(x)
  result = result * x_sign
  # sin(Inf) = NaN, sin(-Inf) = NaN, sin(NaN) = NaN
  return _lazy_map_numbers(d, d.const_like(math.nan), d.const_like(math.nan), d.const_like(math.nan), result)

def xexp2(d:UOp) -> UOp:
  """
  Implements a 1.0 ULP approximation for Ops.EXP2
  - Paper: https://arxiv.org/pdf/2001.09258
  """
  assert d.dtype.scalar() in TRANSCENDENTAL_DTYPES
  # mask +=inf/nan as zero.
  x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  q = rintk(x)
  # s = d - round(d)
  s = x - q.cast(x.dtype)
  # a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
  if d.dtype.scalar() == dtypes.float64:
    u = polyN(s, [0.4434359082926529454e-9, 0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4,
                  0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0,
                  0.6931471805599452862e+0, 0.1000000000000000000e+1])
  else: u = polyN(s, [0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0])
  u = ldexp2k(u, q) # u*2^q
  upper, lower = {dtypes.float64: (1024, -2000), dtypes.float32: (128, -150), dtypes.float16: (23, -22)}[d.dtype.scalar()]
  # Replace x >= upper with +inf
  u = (d >= upper).where(d.const_like(math.inf), u)
  # Replace x < lower with zero.
  u = (d<lower).where(d.const_like(0.0), u)
  # exp2(NaN) = NaN
  return d.ne(d).where(d.const_like(math.nan), u)

def xlog2(d:UOp) -> UOp:
  """
  Implements a 1.0 ULP approximation for Ops.LOG2
  Paper: https://arxiv.org/pdf/2001.09258 5.5
  """
  assert d.dtype.scalar() in TRANSCENDENTAL_DTYPES
  # float16 uses 2^10 for denormal scaling (2^64 overflows), float32/64 use 2^64
  denormal_exp = 10 if d.dtype.scalar() == dtypes.float16 else 64
  FLT_MIN = d.const_like({dtypes.float16: 6.1e-5, dtypes.float32: 1e-4, dtypes.float64: 1e-4}[d.dtype.scalar()])
  is_denormal = d<FLT_MIN
  a = is_denormal.where(d * (2 ** denormal_exp), d)

  e = ilogb2k(a * (1.0 / 0.75)).cast(a.dtype)
  m = ldexp3k(a, -e)
  e = is_denormal.where(e - denormal_exp, e)

  x = (m - 1.0) / (m + 1.0)
  x2 = x * x
  if d.dtype.scalar() == dtypes.float64:
    t = polyN(x2, [0.2211941750456081490e+0, 0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0,
                   0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449])
    r = t * (x * x2) + e + x * 2.885390081777926774
  else:
    t = polyN(x2, [0.4374550283e+0, 0.5764790177e+0, 0.9618012905120])
    # s_lo term (x*3.27e-08) only for float32 - underflows in float16
    r = t * (x * x2) + e + x * 2.8853900432586669922 + (x * 3.2734474483568488616e-08 if d.dtype.scalar() == dtypes.float32 else 0)

  # log2(Inf) = Inf
  r = d.ne(math.inf).where(r, r.const_like(math.inf))
  # log2(0) = -Inf (handle both +0.0 and -0.0)
  r = d.ne(0.0).where(r, r.const_like(-math.inf))
  # log2(x) = NaN for x < 0
  r = (d<-0.0).where(r.const_like(math.nan), r)
  # log2(NaN) = NaN
  r = d.ne(d).where(r.const_like(math.nan), r)
  # log2(-0.0) = -Inf. In certain devices like PTX, x == -0.0 won't be true. so making reciprocal.
  return d.reciprocal().ne(-math.inf).where(r, r.const_like(-math.inf))

def xpow(base:UOp, exponent:UOp) -> UOp:
  # start with b ** e = exp2(e * log2(b))
  ret = (base < 0).where(-base, base).log2().mul(exponent).exp2()
  # negative base: nan for non-integer exponent, negate for odd integer exponent
  non_int = exponent != exponent.cast(dtypes.int32).cast(exponent.dtype)
  is_odd = (exponent < 0).where(-exponent, exponent).cast(dtypes.int32).mod(2).cast(dtypes.bool)
  neg_base = non_int.where(ret.const_like(math.nan), is_odd.where(-ret, ret))
  # fix 0 ** 0 = 1
  return (base.eq(0) & exponent.eq(0)).where(ret.const_like(1), (base < 0).where(neg_base, ret))

# *** integer division ***

@functools.lru_cache(None)
def magicgu(vmax:int, d:int) -> tuple[int,int]:
  # calculate m,s such that x//d == (x*m) >> s for all 0 <= x <= vmax, d>0; adapted from Hacker's Delight, Chapter 10
  nc = (vmax+1)//(d) * d - 1
  nbits = vmax.bit_length()
  for s in range(0, 2*nbits + 1):
    if 2**s > nc*(d - 1 - (2**s - 1) % d):
      m = (2**s + d - 1 - (2**s - 1) % d)//d
      return m, s
  assert False

def fast_idiv(device: str, x: UOp, d: int, dont_cast=False) -> UOp|None:
  # NOTE: disable for METAL due to compiler bug. keccak with -O0 works but not with optimization
  if device.startswith("METAL"): return None
  # If d is a power of two this is not valid for signed ints!
  is_unsigned = x.vmin>=0 or x.dtype in dtypes.uints
  assert d>0, "Sign should have been taken out of divisor"
  vmin,vmax = max(x.vmin, x.dtype.min), min(x.vmax, x.dtype.max)
  m,s = magicgu(max(vmax, abs(vmin)), d)
  if m*vmin >= dtypes.min(x.dtype) and m*vmax <= dtypes.max(x.dtype):
    return ((x*m) >> s) if is_unsigned else ((x*m) >> s) + (x<0).where(x.ufix(1), 0)
  # before we try casting to a larger dtype (slow), we see if there are powers of two in d we can shift to make x smaller
  if (largest_factor_of_two_in_d := (d & -d)) > 1:
    if (ret:=fast_idiv(device, x//largest_factor_of_two_in_d, d//largest_factor_of_two_in_d, dont_cast=True)) is not None: return ret
  if dont_cast: return None
  # promo_lattice needs to return an unsigned type if the type is unsigned
  if dtypes.is_int(next_dtype := promo_lattice[x.dtype.scalar()][-1]) and is_dtype_supported(next_dtype, device):
    if m*vmin >= dtypes.min(next_dtype) and m*vmax <= dtypes.max(next_dtype):
      return ((x.cast(next_dtype)*m) >> s).cast(x.dtype) if is_unsigned else ((x.cast(next_dtype)*m) >> s).cast(x.dtype) + (x<0).where(x.ufix(1), 0)
  return None

# ***** threefry *****

def threefry2x32(x: UOp, key: UOp):
  # split x and key from uint64 to two uint32
  x0, x1 = (x & 0xffffffff).cast(dtypes.uint32), ((x // 2**32) & 0xffffffff).cast(dtypes.uint32)
  key0, key1 = (key & 0xffffffff).cast(dtypes.uint32), ((key // 2**32) & 0xffffffff).cast(dtypes.uint32)

  rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  ks = [key1, key0 ^ key1 ^ 0x1BD11BDA, key0]
  xr:list[UOp] = [x0 + ks[-1], x1 + ks[0]]
  for i in range(5):
    for r in rotations[i % 2]: xr[0], xr[1] = (x0 := xr[0] + xr[1]), x0 ^ ((xr[1] * 2**r) + (xr[1] // 2**(32 - r)))
    xr = [(xr[0] + ks[i % 3]), (xr[1] + ks[(i + 1) % 3] + i + 1)]

  return xr[1].cast(dtypes.uint64) * 2**32 | xr[0].cast(dtypes.uint64)

# ***** long as 2 ints *****

l2i_dt = {dtypes.long: dtypes.int, dtypes.ulong: dtypes.uint}
def unpack32(v:UOp) -> tuple[UOp, UOp]: return v.bitcast(dtypes.uint) & 0xFFFF, v.bitcast(dtypes.uint) >> 16
def l2i_idx(idx:UOp, off:int) -> UOp: return idx.replace(src=(idx.src[0], idx.src[1]*2+off))

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
    case Ops.CAST if dt == dtypes.bool: return a0.ne(UOp.const(a0.dtype, 0)) | a1.ne(UOp.const(a1.dtype, 0))
    case Ops.CAST: return a0.bitcast(dtypes.uint).cast(dt)
    case Ops.BITCAST: return a0.bitcast(dt), a1.bitcast(dt)
    case Ops.SHL:
      lo, hi = a0 << (b0_mod:=b0 & 31), (a1 << b0_mod) | ((a0 >> 1) >> (31 - b0_mod))
      return (b0 >= 32).where(zero, lo), (b0 >= 32).where(lo, hi)
    case Ops.SHR:
      lo, hi = (a0 >> (b0_mod:=b0 & 31)) | ((a1 << 1) << (31 - b0_mod)), a1 >> b0_mod
      return (b0 >= 32).where(hi, lo), (b0 >= 32).where(zero, hi)
    case Ops.ADD: return (low:=a0+b0), (a1 + b1).replace(dtype=dt) + (low.bitcast(dtypes.uint) < a0.bitcast(dtypes.uint)).cast(dt)
    case Ops.SUB: return a0 - b0, a1 - b1 - (a0.bitcast(dtypes.uint) < b0.bitcast(dtypes.uint)).cast(dt)
    case Ops.MUL:
      (a00, a01), (b00, b01) = unpack32(a0), unpack32(b0)
      mid = l2i(Ops.ADD, dt, ((a00*b01)<<16).bitcast(dt), ((a00*b01)>>16).bitcast(dt), ((a01*b00)<<16).bitcast(dt), ((a01*b00)>>16).bitcast(dt))
      return l2i(Ops.ADD, dt, *mid, (a00*b00).bitcast(dt), (a01*b01).bitcast(dt) + a0*b1 + a1*b0)
    case Ops.IDIV | Ops.MOD:
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
        q = ((q[0] | cond.cast(dtypes.uint) << (i % 32), q[1]) if i < 32 else (q[0], q[1] | cond.cast(dtypes.uint) << (i % 32)))
        r = l2i(Ops.WHERE, dtypes.uint, cond, *diff, *r)
      if dt == dtypes.int:
        (nq0, nq1), (nr0, nr1) = l2i(Ops.BITCAST, dt, *l2i(Ops.NEG, dtypes.uint, *q)), l2i(Ops.BITCAST, dt, *l2i(Ops.NEG, dtypes.uint, *r))
        (q0, q1), (r0, r1) = l2i(Ops.BITCAST, dt, *q), l2i(Ops.BITCAST, dt, *r)
        return (a_neg.where(nr0, r0), a_neg.where(nr1, r1)) if op == Ops.MOD else ((a_neg^b_neg).where(nq0, q0), (a_neg^b_neg).where(nq1, q1))
      return (r[0].bitcast(dt), r[1].bitcast(dt)) if op == Ops.MOD else (q[0].bitcast(dt), q[1].bitcast(dt))
    case Ops.CMPLT: return (a1 < b1) | ((a1.eq(b1)) & (a0.bitcast(dtypes.uint) < b0.bitcast(dtypes.uint)))
    case Ops.CMPEQ: return a0.eq(b0) & a1.eq(b1)
    case Ops.CMPNE: return a0.ne(b0) | a1.ne(b1)
    case Ops.XOR | Ops.OR | Ops.AND: return UOp(op, dt, src=(a0, b0)), UOp(op, dt, src=(a1, b1))
    case Ops.WHERE: return uops[0].where(uops[1], uops[3]), uops[0].where(uops[2], uops[4])
    case Ops.MAX: return l2i(Ops.WHERE, dt, l2i(Ops.CMPLT, dt, *uops), b0, b1, a0, a1)
    case _: raise NotImplementedError(f"long decomposition of {op} unsupported")

# ***** decomposition patterns *****

powers_of_two: dict[int, int] = {2**i:i for i in range(64)}
@functools.cache
def get_late_rewrite_patterns(ops:tuple[Ops, ...], device:str, force_transcendental:bool, disable_fast_idiv:bool,
                              emulated_dtypes:tuple[DType, ...]) -> PatternMatcher:
  pat: list[tuple[UPat, Callable]] = []
  for op,f in ((Ops.EXP2, xexp2), (Ops.LOG2, xlog2), (Ops.SIN, xsin)):
    if op not in ops or force_transcendental:
      pat += [(UPat(op, dtype=TRANSCENDENTAL_DTYPES, src=(UPat.var("d"),)), f),
              (UPat(op, dtype=tuple(dt for dt in dtypes.floats if dt not in TRANSCENDENTAL_DTYPES), src=(UPat.var("d"),), name="x"),
                lambda x,d: d.cast(dtypes.float32).alu(x.op).cast(x.dtype))]
  # rewrite SQRT to xpow 0.5
  if Ops.SQRT not in ops or force_transcendental: pat.append((UPat(Ops.SQRT, src=UPat.var("d")), lambda d: xpow(d, d.const_like(0.5))))
  # no real hardware supports THREEFRY, but NullRenderer does
  if Ops.THREEFRY not in ops: pat.append((UPat(Ops.THREEFRY, dtype=dtypes.uint64, src=(UPat.var("x"), UPat.var("key"))), threefry2x32))
  # MAX can be rewritten as CMPLT + WHERE (max function is annoying on many cstyle backends)
  if Ops.MAX not in ops and Ops.CMPLT in ops: pat.append((UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])))
  # rewrite MOD to AND (which should always be supported, but not for generic in tests): x % (2**y) -> x & (2**y-1)
  if Ops.AND in ops: pat += [(UPat.var("x", dtypes.ints)%UPat.cvar("c"), lambda x,c: x & (c.arg-1) if c.arg in powers_of_two else None)]
  if Ops.OR in ops: pat += [(UPat.var("x", dtypes.bool).logical_not()&UPat.var("y", dtypes.bool).logical_not(),
    lambda x,y: (x | y).logical_not())]
  # rewrite MUL/IDIV to SHL+SHR: x*(2**y) -> shl(x,y) and x//(2**y) -> shr(x,y)
  if Ops.SHL in ops: pat += [(UPat.var("x", dtypes.ints)*UPat.cvar("c"), lambda c,x: x << v if (v:=powers_of_two.get(c.arg, 0)) else None)]
  if Ops.SHR in ops:
    # no reason to check x<0 for uints
    pat += [(UPat.var("x", dtypes.uints)//UPat.cvar("c"), lambda x,c: x >> v if (v:=powers_of_two.get(c.arg, 0)) else None)]
    pat += [(UPat.var("x", dtypes.ints)//UPat.cvar("c"), lambda x,c: (x+(l.const_like(l.vmin) if (l:=(x<0)).vmin==l.vmax else l).where(
      c-1, 0)) >> v if (v:=powers_of_two.get(c.arg, 0)) else None)]  # (x+(x<0).where(c-1, 0)) >> v
    if not disable_fast_idiv:
      pat += [(UPat.var("x", dtypes.ints)//UPat.cvar("d", vec=False), lambda ctx, x, d: fast_idiv(ctx, x, d.arg))]
      pat += [(UPat.var("x", dtypes.ints)%UPat.var("d"), lambda x, d: x-d*(x//d))]
  if Ops.NEG in ops:
    pat += [(UPat.var('x')*-1, lambda ctx,x: x.alu(Ops.NEG))]
    if Ops.SUB in ops: pat += [(UPat.var('x')+UPat.var('y').alu(Ops.NEG), lambda ctx,x,y: x.alu(Ops.SUB, y))]
  if Ops.CMPLT in ops:
    # These are late rewrites because simplex expects equalities to be a certain format
    pat += [
      ((UPat.var("x", dtypes.sints) < UPat.cvar("c", dtypes.sints)).logical_not(), lambda x,c: c-1<x),
      ((UPat.cvar("c", dtypes.sints) < UPat.var("x", dtypes.sints)).logical_not(), lambda x,c: x<c+1),
      (UPat.var("x", dtypes.sints)*-1 < UPat.var("y", dtypes.sints)*UPat.cvar("c"), lambda x,y,c: y*(-c)<x),
      (UPat.var("x", dtypes.sints)*-1 < UPat.cvar("c"), lambda x,c:-c<x),
      ((UPat.cvar("c1",vec=False)<UPat.var("x", dtypes.sints)) & (UPat.var("x", dtypes.sints)<UPat.cvar("c2",vec=False)),
        lambda x,c1,c2: x.eq(c1+1) if c1.arg+1==c2.arg-1 else None),  # (c-1)<x & x<(c+1) -> x==c
    ]
  if Ops.CMPEQ in ops: pat += [(UPat.var('x').ne(UPat.var('y')).logical_not(), lambda x,y: x.alu(Ops.CMPEQ, y))]
  if Ops.MULACC in ops: pat += [(UPat.var('a')*UPat.var('b')+UPat.var('c'), lambda a,b,c: a.alu(Ops.MULACC, b, c))]
  # some backends emit FDIV for RECIP, in that case: a*(1/b) -> a/b
  if Ops.FDIV in ops:
    pat += [(UPat.var("x").reciprocal(), lambda x: x.const_like(1).alu(Ops.FDIV, x))]
    pat += [(UPat.var("a", dtypes.floats) * UPat.const(dtypes.floats, 1).alu(Ops.FDIV, UPat.var("b")), lambda a,b: a.alu(Ops.FDIV, b))]
  if not is_dtype_supported(dtypes.long, device) or dtypes.long in emulated_dtypes:
    pat += [(UPat((*GroupOp.Defines, Ops.INDEX), name="x"), lambda x:
             x.replace(dtype=l2i_dt[x.dtype.base].ptr(x.dtype.size * 2)) if hasattr(x.dtype, 'size') and x.dtype.base in l2i_dt else None)]
    pat += [(UPat(Ops.INDEX, tuple(l2i_dt.keys()), name='x'), lambda x:
             None if x.tag is None else x.replace(dtype=l2i_dt[x.dtype], src=(x.src[0], x.src[1]*2+x.tag)))]
    pat += [(UPat(Ops.STORE, src=(UPat.var('idx'), UPat.var('val', tuple(l2i_dt.keys()))), name='st'), lambda st,idx,val:
             st.replace(src=(l2i_idx(idx, 0), val.rtag(0))).group(st.replace(src=(l2i_idx(idx, 1), val.rtag(1)))) if val.tag is None else None)]
    pat += [(UPat(GroupOp.Comparison, src=(UPat.var('a', tuple(l2i_dt.keys())), UPat.var('b', tuple(l2i_dt.keys()))), name="x"), lambda a,b,x:
             l2i(x.op, dt:=l2i_dt[a.dtype], a.rtag(0).cast(dt), a.rtag(1).cast(dt), b.rtag(0).cast(dt), b.rtag(1).cast(dt)))]
    pat += [(UPat(Ops.CAST, tuple(l2i_dt.keys()), src=(UPat.var('a'),), name="x"), lambda a,x:
             l2i(x.op, x.dtype, a)[x.tag] if x.tag is not None and a.dtype not in l2i_dt else None)]
    pat += [(UPat(Ops.CAST, tuple(l2i_dt.keys()), src=(UPat.var('a', tuple(l2i_dt.keys())),), name="x"), lambda a,x:
             None if x.tag is None else (a.rtag(0).cast(dt:=l2i_dt[a.dtype]).bitcast(xdt:=l2i_dt[x.dtype]), a.rtag(1).cast(dt).bitcast(xdt))[x.tag])]
    pat += [(UPat(Ops.CAST, src=(UPat.var('a', tuple(l2i_dt.keys())),), name="x"), lambda a,x:
             l2i(x.op, x.dtype, a.rtag(0).cast(dt:=l2i_dt[a.dtype]), a.rtag(1).cast(dt)) if x.dtype not in l2i_dt and a.tag is None else None)]
    pat += [(UPat((*(GroupOp.ALU - GroupOp.Comparison), Ops.BITCAST), tuple(l2i_dt.keys()), name="x"), lambda x:
             None if x.tag is None else l2i(x.op, l2i_dt[x.dtype], *flatten((a.rtag(0).cast(dt:=l2i_dt[x.src[-1].dtype]), a.rtag(1).cast(dt))
                                                                            if a.dtype in l2i_dt else (a,) for a in x.src))[x.tag])]
    pat += [(UPat(Ops.LOAD, tuple(l2i_dt.keys()), src=(UPat.var('idx'),), name='x'), lambda x,idx:
             None if x.tag is None else x.replace(dtype=l2i_dt[x.dtype], src=(l2i_idx(idx, x.tag),)))]
    pat += [(UPat(Ops.CONST, tuple(l2i_dt.keys()), name='x'), lambda x:
             None if x.tag is None else UOp.const(dt:=l2i_dt[x.dtype], truncate[dt]((x.arg >> 32) if x.tag == 1 else (x.arg & 0xFFFFFFFF))))]
  return PatternMatcher(pat)
