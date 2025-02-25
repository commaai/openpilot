import math
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import polyN
from tinygrad.ops import UOp

TRANSCENDENTAL_SUPPORTED_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64)

def _lazy_map_numbers(x:UOp, inf:UOp, _inf:UOp, nan:UOp, ratio:UOp):
  """replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio"""
  return x.ne(math.inf).where(x.ne(x).where(nan, x.ne(-math.inf).where(ratio, _inf)), inf)

# *** helper functions for bit manipulation ***
def mantissa_bits(d:DType) -> int: return dtypes.finfo(d)[1]
def exponent_bias(d:DType) -> int: return {dtypes.float64: 1023, dtypes.float32: 127, dtypes.float16: 15}[d]
def exponent_mask(d:DType) -> int: return {dtypes.float64: 2047, dtypes.float32: 255, dtypes.float16: 31}[d]

# **** utils ****
def shr(x:UOp, y:int) -> UOp: return x // (2**y)
def shl(x:UOp, y:int) -> UOp: return x * (2**y)

def rintk(d:UOp) -> UOp:
  """round d:float to int away from 0"""
  out_dtype = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype]
  return (d + (d<0.0).where(d.const_like(-0.5), d.const_like(0.5))).cast(out_dtype)

def pow2if(q:UOp, float_dtype:DType):
  """cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]"""
  out_dtype = {dtypes.int64: dtypes.float64, dtypes.int32: dtypes.float32, dtypes.int16: float_dtype}[q.dtype]
  return shl(q + exponent_bias(out_dtype), mantissa_bits(out_dtype)).bitcast(out_dtype)

def ilogb2k(d:UOp) -> UOp:
  """calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf)."""
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  dint = d.bitcast({dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype])
  # -1 <= ilog2bk(d) <= 128
  return (shr(dint, mantissa_bits(d.dtype)) & exponent_mask(d.dtype)) - exponent_bias(d.dtype)

def ldexp3k(d:UOp, e:UOp) -> UOp:
  """d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number."""
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES and e.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  cast_map = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}
  m1 = d.bitcast(cast_map[d.dtype])
  m2 = shl(e.cast(cast_map[d.dtype]), mantissa_bits(d.dtype))
  return (m1 + m2).bitcast(d.dtype).cast(d.dtype)

def ldexp2k(d:UOp, e:UOp) -> UOp:
  """d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal."""
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES and e.dtype in (dtypes.int16, dtypes.int32, dtypes.int64)
  return (d * pow2if(shr(e, 1), d.dtype)) * pow2if(e - shr(e, 1), d.dtype)

def frexp(v:UOp) -> tuple[UOp, UOp]:
  """frexp(v) -> (mantissa, exponent) assuming v != 0"""
  assert v.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  # m1 = masks for mantissa, m2 = masks to normalize the mantissa.
  m1 = {dtypes.float64: 0x000FFFFFFFFFFFFF, dtypes.float32: 0x807FFFFF, dtypes.float16: 0x83FF}[v.dtype]
  m2 = {dtypes.float64: 0x3FE0000000000000, dtypes.float32: 0x3F000000, dtypes.float16: 0x3800}[v.dtype]
  bits = v.bitcast({dtypes.float64: dtypes.uint64, dtypes.float32: dtypes.uint32, dtypes.float16: dtypes.uint16}[v.dtype])
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
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  # https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c/30465751#30465751
  # 190 bits of 2/pi for Payne-Hanek style argument reduction
  two_over_pi_f = [0x00000000, 0x28be60db, 0x9391054a, 0x7f09d5f4, 0x7d4d3770, 0x36d8a566, 0x4f10e410]

  intermediate_dtype = dtypes.float32 if d.dtype == dtypes.float16 else d.dtype

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
  def _shl_lazy(x, y): return (x.cast(dtypes.uint64) * pow2if(y, d.dtype).cast(dtypes.uint64)).cast(dtypes.uint32)
  def _shr_lazy(x, y): return (x.cast(dtypes.uint64) // pow2if(y, d.dtype).cast(dtypes.uint64)).cast(dtypes.uint32)

  a = [_take(UOp.const(dtypes.uint32, 0), i) for i in range(4)]
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
    if x.dtype == dtypes.float64:
      # https://github.com/shibatch/sleef/blob/f6d8a841fbfddd26ce712834d4da220cd76048fb/src/common/misc.h#L77
      PI_A, PI_B, PI_C, PI_D = 3.1415926218032836914, 3.1786509424591713469e-08, 1.2246467864107188502e-16, 1.2736634327021899816e-24
      d = qdh * -PI_A + x
      d = q * -PI_A + d
      d = qdh * -PI_B + d
      d = q * -PI_B + d
      d = qdh * -PI_C + d
      d = q * -PI_C + d
      d = (qdh + q) * -PI_D + d
    elif x.dtype == dtypes.float16:
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
  quadrant = rintk(d * m_1_pi -qdh) if d.dtype == dtypes.float64 else rintk(d * m_1_pi)
  return _reduce_d(d, quadrant.cast(d.dtype)), quadrant.cast(dtypes.int32)

# *** approximate sine on small angle. ***
def trig_poly(d:UOp, coeff32, coeff64): return d * (polyN(d*d, coeff64) if d.dtype == dtypes.float64 else polyN(d*d, coeff32))
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
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
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
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  # mask +=inf/nan as zero.
  x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  q = rintk(x)
  # s = d - round(d)
  s = x - q.cast(x.dtype)
  # a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
  if d.dtype == dtypes.float64:
    u = polyN(s, [0.4434359082926529454e-9, 0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4,
                  0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0,
                  0.6931471805599452862e+0, 0.1000000000000000000e+1])
  else: u = polyN(s, [0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0])
  u = ldexp2k(u, q) # u*2^q
  upper, lower = {dtypes.float64: (1024, -2000), dtypes.float32: (128, -150), dtypes.float16: (23, -22)}[d.dtype]
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
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  # TODO: float16 denormal need float32 to achieve precision
  if d.dtype == dtypes.float16: return xlog2(d.cast(dtypes.float32)).cast(dtypes.float16)
  FLT_MIN = d.const_like(1e-6 if d.dtype == dtypes.float16 else 1e-4)
  is_denormal = d<FLT_MIN
  a = is_denormal.where(d * (2 ** 64), d)

  e = ilogb2k(a * (1.0 / 0.75)).cast(a.dtype)
  m = ldexp3k(a, -e)
  e = is_denormal.where(e - 64, e)

  x = (m - 1.0) / (m + 1.0)
  x2 = x * x
  if d.dtype == dtypes.float64:
    t = polyN(x2, [0.2211941750456081490e+0, 0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0,
                   0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449])
    s_hi, s_lo = e+x*2.885390081777926774, e.const_like(0)
  else:
    t = polyN(x2, [0.4374550283e+0, 0.5764790177e+0, 0.9618012905120])
    s_hi, s_lo = e+x*2.8853900432586669922, x*3.2734474483568488616e-08
  r = t * (x * x2) + (s_hi + s_lo)

  # log2(Inf) = Inf
  r = d.ne(math.inf).where(r, r.const_like(math.inf))
  # log2(x) = NaN for x < 0
  r = (d<-0.0).where(r.const_like(math.nan), r)
  # log2(0) = -Inf, but we will compare using the value of y because 1e-200==0 is true.
  # log2_zero = the value of unmasked xlog2(0.0).
  log2_zero = {dtypes.float64: -1087, dtypes.float32: -191, dtypes.float16: -79}[d.dtype]
  r = r.ne(log2_zero).where(r, r.const_like(-math.inf))
  # log2(NaN) = NaN
  r = d.ne(d).where(r.const_like(math.nan), r)
  # log2(-0.0) = -Inf. In certain devices like PTX, x == -0.0 won't be true. so making reciprocal.
  return d.reciprocal().ne(-math.inf).where(r, r.const_like(-math.inf))

def xpow(base:UOp, exponent:UOp) -> UOp:
  # start with b ** e = exp2(e * log2(b))
  ret = (base < 0).where(-base, base).log2().mul(exponent).exp2()
  # negative base adjustment: nan for non-integer exponent and -1 for odd exponent
  non_int = exponent != exponent.cast(dtypes.int32).cast(exponent.dtype)
  adj = non_int.where(ret.const_like(math.nan),
    (exponent < 0).where(-exponent, exponent).cast(dtypes.int32).mod(2).cast(dtypes.bool).where(ret.const_like(-1), ret.const_like(1)))
  # fix 0 ** 0 = 1
  return (base.eq(0) & exponent.eq(0)).where(ret.const_like(1), ret * (base < 0).where(adj, ret.const_like(1)))
