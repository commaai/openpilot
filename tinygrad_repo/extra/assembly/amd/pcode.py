# Tokenizer-based expression parser for AMD pcode
from typing import Any, Callable
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

# Type alias for vars dict: stores UOps for variables and tuples for lambda definitions
VarVal = UOp | tuple[str, list[str], str]

def _const(dt, v): return UOp.const(dt, v)
def _u32(v): return _const(dtypes.uint32, v)
def _u64(v): return _const(dtypes.uint64, v)
def _to_u32(v): return v if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)
def _to_bool(v): return v if v.dtype == dtypes.bool else v.ne(_const(v.dtype, 0))
def _cast_to(v, dt):
  if v.dtype == dt: return v
  if dt == dtypes.half: return v.cast(dtypes.uint16).bitcast(dtypes.half)
  return v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)

# Float bit extraction - returns (bits, exp_mask, mant_mask, quiet_bit, exp_shift) based on float type
def _float_info(v: UOp) -> tuple[UOp, UOp, UOp, UOp, int]:
  if v.dtype in (dtypes.float64, dtypes.uint64):
    bits = v.bitcast(dtypes.uint64) if v.dtype == dtypes.float64 else v.cast(dtypes.uint64)
    return bits, _u64(0x7FF0000000000000), _u64(0x000FFFFFFFFFFFFF), _u64(0x0008000000000000), 52
  if v.dtype in (dtypes.half, dtypes.uint16):
    bits = (v.bitcast(dtypes.uint16) if v.dtype == dtypes.half else (v & _u32(0xFFFF)).cast(dtypes.uint16)).cast(dtypes.uint32)
    return bits, _u32(0x7C00), _u32(0x03FF), _u32(0x0200), 10
  bits = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v.cast(dtypes.uint32)
  return bits, _u32(0x7F800000), _u32(0x007FFFFF), _u32(0x00400000), 23

def _isnan(v: UOp) -> UOp:
  bits, exp_m, mant_m, _, _ = _float_info(v.cast(dtypes.float32) if v.dtype == dtypes.half else v)
  return (bits & exp_m).eq(exp_m) & (bits & mant_m).ne(_const(bits.dtype, 0))

def _bitreverse(v: UOp, bits: int) -> UOp:
  dt, masks = (dtypes.uint64, [(0x5555555555555555,1),(0x3333333333333333,2),(0x0F0F0F0F0F0F0F0F,4),(0x00FF00FF00FF00FF,8),(0x0000FFFF0000FFFF,16)]) \
    if bits == 64 else (dtypes.uint32, [(0x55555555,1),(0x33333333,2),(0x0F0F0F0F,4),(0x00FF00FF,8)])
  v = v.cast(dt) if v.dtype != dt else v
  for m, s in masks: v = ((v >> _const(dt, s)) & _const(dt, m)) | ((v & _const(dt, m)) << _const(dt, s))
  return (v >> _const(dt, 32 if bits == 64 else 16)) | (v << _const(dt, 32 if bits == 64 else 16))

def _extract_bits(val: UOp, hi: int, lo: int) -> UOp:
  dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
  return ((val >> _const(dt, lo)) if lo > 0 else val) & _const(val.dtype, (1 << (hi - lo + 1)) - 1)

def _set_bit(old, pos, val):
  mask = _u32(1) << pos
  return (old & (mask ^ _u32(0xFFFFFFFF))) | ((val.cast(dtypes.uint32) & _u32(1)) << pos)

def _val_to_bits(val):
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.float64: return val.bitcast(dtypes.uint64)
  return val if val.dtype == dtypes.uint32 else val.cast(dtypes.uint32)

def _floor(x): t = UOp(Ops.TRUNC, x.dtype, (x,)); return ((x < _const(x.dtype, 0)) & x.ne(t)).where(t - _const(x.dtype, 1), t)
def _f16_extract(v): return (v & _u32(0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half) if v.dtype == dtypes.uint32 else v

def _check_nan(v: UOp, quiet: bool) -> UOp:
  if v.op == Ops.CAST and v.dtype == dtypes.float64: v = v.src[0]
  bits, exp_m, mant_m, qb, _ = _float_info(v)
  is_nan_exp, has_mant, is_q = (bits & exp_m).eq(exp_m), (bits & mant_m).ne(_const(bits.dtype, 0)), (bits & qb).ne(_const(bits.dtype, 0))
  return (is_nan_exp & is_q) if quiet else (is_nan_exp & has_mant & is_q.logical_not())

def _minmax_reduce(is_max: bool, dt, *args: UOp) -> UOp:
  def cast(v: UOp) -> UOp: return v.bitcast(dt) if dt == dtypes.float32 and v.dtype == dtypes.uint32 else v.cast(dt)
  def minmax(a: UOp, b: UOp) -> UOp:
    if dt in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64): return (a > b).where(a, b) if is_max else (a < b).where(a, b)
    return a.maximum(b) if is_max else a.minimum(b)
  result = cast(args[0])
  for a in args[1:]:
    b = cast(a)
    if dt == dtypes.float32: result = _isnan(result).where(b, _isnan(b).where(result, minmax(result, b)))
    else: result = minmax(result, b)
  return result

def _find_two_pi_mul(x):
  if x.op != Ops.MUL or len(x.src) != 2: return None
  for i, s in enumerate(x.src):
    if s.op == Ops.CONST and abs(s.arg - 6.283185307179586) < 1e-5: return (x.src[1-i], 6.283185307179586)
    if s.op == Ops.MUL and len(s.src) == 2:
      vals = [ss.arg for ss in s.src if ss.op == Ops.CONST] + [ss.src[0].arg for ss in s.src if ss.op == Ops.CAST and ss.src[0].op == Ops.CONST]
      if len(vals) == 2 and abs(vals[0] * vals[1] - 6.283185307179586) < 1e-5: return (x.src[1-i], vals[0] * vals[1])
  return None

def _trig_reduce(x, phase=0.0):
  match = _find_two_pi_mul(x)
  if match is not None:
    turns, two_pi = match
    if phase: turns = turns + _const(turns.dtype, phase)
    n = _floor(turns + _const(turns.dtype, 0.5))
    return UOp(Ops.SIN, turns.dtype, ((turns - n) * _const(turns.dtype, two_pi),))
  if phase: x = x + _const(x.dtype, phase * 6.283185307179586)
  n = _floor(x * _const(x.dtype, 0.15915494309189535) + _const(x.dtype, 0.5))
  return UOp(Ops.SIN, x.dtype, (x - n * _const(x.dtype, 6.283185307179586),))

def _signext(val: UOp) -> UOp:
  for bits, mask, ext in [(4, 0xF, 0xFFFFFFF0), (8, 0xFF, 0xFFFFFF00), (16, 0xFFFF, 0xFFFF0000)]:
    if (val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == mask) or val.dtype.itemsize == bits // 8:
      v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
      sb = (v32 >> _u32(bits - 1)) & _u32(1)
      return sb.ne(_u32(0)).where(v32 | _u32(ext), v32).cast(dtypes.int)
  return val.cast(dtypes.int64) if val.dtype in (dtypes.int, dtypes.int32) else val

def _signext_4bit(val: UOp) -> UOp:
  """Sign extend a 4-bit value to 32-bit signed integer."""
  v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  sb = (v32 >> _u32(3)) & _u32(1)  # sign bit at position 3
  return sb.ne(_u32(0)).where(v32 | _u32(0xFFFFFFF0), v32).bitcast(dtypes.int)

def _abs(val: UOp) -> UOp:
  if val.dtype not in (dtypes.float32, dtypes.float64, dtypes.half): return val
  _, _, _, _, shift = _float_info(val)
  sign_mask = {10: 0x7FFF, 23: 0x7FFFFFFF, 52: 0x7FFFFFFFFFFFFFFF}[shift]
  bt, ft = {10: (dtypes.uint16, dtypes.half), 23: (dtypes.uint32, dtypes.float32), 52: (dtypes.uint64, dtypes.float64)}[shift]
  return (val.bitcast(bt) & _const(bt, sign_mask)).bitcast(ft)

def _f_to_u(f, dt): return UOp(Ops.TRUNC, f.dtype, ((f < _const(f.dtype, 0.0)).where(_const(f.dtype, 0.0), f),)).cast(dt)

def _cvt_quiet(val: UOp) -> UOp:
  bits, _, _, qb, _ = _float_info(val)
  bt, ft = (dtypes.uint64, dtypes.float64) if val.dtype == dtypes.float64 else (dtypes.uint16, dtypes.half) if val.dtype == dtypes.half else (dtypes.uint32, dtypes.float32)
  return (val.bitcast(bt) | qb).bitcast(ft)

def _is_denorm(val: UOp) -> UOp:
  bits, exp_m, mant_m, _, _ = _float_info(val)
  return (bits & exp_m).eq(_const(bits.dtype, 0)) & (bits & mant_m).ne(_const(bits.dtype, 0))

_EXP_BITS = {10: 0x1F, 23: 0xFF, 52: 0x7FF}
def _get_exp(bits: UOp, shift: int) -> UOp: return ((bits >> _const(bits.dtype, shift)) & _const(bits.dtype, _EXP_BITS[shift])).cast(dtypes.int)

def _exponent(val: UOp) -> UOp:
  bits, _, _, _, shift = _float_info(val)
  return _get_exp(bits, shift)

def _div_would_be_denorm(a: UOp, b: UOp) -> UOp:
  bits_n, _, _, _, shift = _float_info(a)
  bits_d, _, _, _, _ = _float_info(b)
  min_exp = {10: -14, 23: -126, 52: -1022}[shift]
  return (_get_exp(bits_n, shift) - _get_exp(bits_d, shift)) < _const(dtypes.int, min_exp)

def _sign(val: UOp) -> UOp:
  bits, _, _, _, shift = _float_info(val)
  sign_shift = {10: 15, 23: 31, 52: 63}[shift]
  return ((bits >> _const(bits.dtype, sign_shift)) & _const(bits.dtype, 1)).cast(dtypes.uint32)

def _signext_from_bit(val: UOp, w: UOp) -> UOp:
  is_64bit = val.dtype in (dtypes.uint64, dtypes.int64)
  dt = dtypes.uint64 if is_64bit else dtypes.uint32
  mask_all = _const(dt, 0xFFFFFFFFFFFFFFFF if is_64bit else 0xFFFFFFFF)
  one = _const(dt, 1)
  val_u = val.cast(dt) if val.dtype != dt else val
  w_val = w.cast(dt) if w.dtype != dt else w
  sign_bit = (val_u >> (w_val - one)) & one
  ext_mask = ((one << w_val) - one) ^ mask_all
  return sign_bit.ne(_const(dt, 0)).where(val_u | ext_mask, val_u)

def _ldexp(val: UOp, exp: UOp) -> UOp:
  if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
  elif val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
  if exp.dtype in (dtypes.uint32, dtypes.uint64): exp = exp.cast(dtypes.int if exp.dtype == dtypes.uint32 else dtypes.int64)
  return val * UOp(Ops.EXP2, val.dtype, (exp.cast(val.dtype),))

def _frexp_mant(val: UOp) -> UOp:
  val = val.bitcast(dtypes.float32) if val.dtype == dtypes.uint32 else val.bitcast(dtypes.float64) if val.dtype == dtypes.uint64 else val
  if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) & _u32(0x807FFFFF)) | _u32(0x3f000000)).bitcast(dtypes.float32)
  return ((val.bitcast(dtypes.uint64) & _const(dtypes.uint64, 0x800FFFFFFFFFFFFF)) | _const(dtypes.uint64, 0x3fe0000000000000)).bitcast(dtypes.float64)

def _frexp_exp(val: UOp) -> UOp:
  val = val.bitcast(dtypes.float32) if val.dtype == dtypes.uint32 else val.bitcast(dtypes.float64) if val.dtype == dtypes.uint64 else val
  if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) >> _u32(23)) & _u32(0xFF)).cast(dtypes.int) - _const(dtypes.int, 126)
  return ((val.bitcast(dtypes.uint64) >> _const(dtypes.uint64, 52)) & _const(dtypes.uint64, 0x7FF)).cast(dtypes.int) - _const(dtypes.int, 1022)

TWO_OVER_PI = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6
# TWO_OVER_PI as 19 u64 words for trig_preop_result (word[0] = bits 0-63, word[18] = bits 1152-1200)
_PREOP_WORDS = tuple((TWO_OVER_PI >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(19))
def _trig_preop(val: UOp) -> UOp:
  # Extract 53 bits from position (1148 - shift) in the 1201-bit 2/PI constant
  # Using word-based selection: 19 conditions instead of 1149
  shift = val.cast(dtypes.uint32)
  bit_pos = _u32(1148) - shift  # starting bit position from LSB
  word_idx = bit_pos >> _u32(6)  # // 64
  bit_off = bit_pos & _u32(63)   # % 64
  # Select lo_word and hi_word using shared conditions
  lo_word, hi_word = _u64(_PREOP_WORDS[18]), _u64(0)
  for i in range(17, -1, -1):
    cond = word_idx.eq(_u32(i))
    lo_word = cond.where(_u64(_PREOP_WORDS[i]), lo_word)
    hi_word = cond.where(_u64(_PREOP_WORDS[i + 1]), hi_word)
  # Combine and extract 53 bits: ((lo >> bit_off) | (hi << (64 - bit_off))) & mask
  bit_off_64 = bit_off.cast(dtypes.uint64)
  result = ((lo_word >> bit_off_64) | (hi_word << (_u64(64) - bit_off_64))) & _u64(0x1fffffffffffff)
  return result.cast(dtypes.float64)

def _ff1(val: UOp, bits: int) -> UOp:
  dt = dtypes.uint64 if bits == 64 else dtypes.uint32
  val = val.cast(dt) if val.dtype != dt else val
  result = _const(dtypes.int, -1)
  for i in range(bits):
    cond = ((val >> _const(dt, i)) & _const(dt, 1)).ne(_const(dt, 0)) & result.eq(_const(dtypes.int, -1))
    result = cond.where(_const(dtypes.int, i), result)
  return result

def _sad_u8(a: UOp, b: UOp, acc: UOp, masked: bool = False) -> UOp:
  """Sum of absolute differences of 4 unsigned bytes + accumulator. If masked, skips bytes where a == 0."""
  a, b, acc = a.cast(dtypes.uint32), b.cast(dtypes.uint32), acc.cast(dtypes.uint32)
  result = acc
  for i in range(4):
    a_byte = (a >> _u32(i * 8)) & _u32(0xFF)
    b_byte = (b >> _u32(i * 8)) & _u32(0xFF)
    diff = (a_byte > b_byte).where(a_byte - b_byte, b_byte - a_byte)
    result = result + (a_byte.ne(_u32(0)).where(diff, _u32(0)) if masked else diff)
  return result

_FUNCS: dict[str, Callable[..., UOp]] = {
  'sqrt': lambda a: UOp(Ops.SQRT, a.dtype, (a,)), 'trunc': lambda a: UOp(Ops.TRUNC, a.dtype, (a,)),
  'log2': lambda a: UOp(Ops.LOG2, a.dtype, (a,)), 'sin': lambda a: _trig_reduce(a),
  'cos': lambda a: _trig_reduce(a, 0.25), 'floor': _floor, 'fract': lambda a: a - _floor(a),
  'signext': _signext, 'abs': _abs,
  'isEven': lambda a: (UOp(Ops.TRUNC, a.dtype, (a,)).cast(dtypes.int) & _const(dtypes.int, 1)).eq(_const(dtypes.int, 0)),
  'max': lambda a, b: UOp(Ops.MAX, a.dtype, (a, b)),
  'min': lambda a, b: UOp(Ops.MAX, a.dtype, (a.neg(), b.neg())).neg(),
  'pow': lambda a, b: UOp(Ops.EXP2, dtypes.float32, (b.bitcast(dtypes.float32),)),
  'fma': lambda a, b, c: a * b + c,
  'i32_to_f32': lambda a: a.cast(dtypes.int).cast(dtypes.float32),
  'u32_to_f32': lambda a: a.cast(dtypes.uint32).cast(dtypes.float32),
  'f32_to_i32': lambda a: UOp(Ops.TRUNC, dtypes.float32, (a.bitcast(dtypes.float32),)).cast(dtypes.int),
  'f32_to_u32': lambda a: _f_to_u(a.bitcast(dtypes.float32), dtypes.uint32),
  'f64_to_i32': lambda a: UOp(Ops.TRUNC, dtypes.float64, (a.bitcast(dtypes.float64),)).cast(dtypes.int),
  'f64_to_u32': lambda a: _f_to_u(a.bitcast(dtypes.float64), dtypes.uint32),
  'f16_to_f32': lambda a: _f16_extract(a).cast(dtypes.float32),
  'f32_to_f16': lambda a: a.cast(dtypes.half),
  'f32_to_f64': lambda a: a.bitcast(dtypes.float32).cast(dtypes.float64),
  'f64_to_f32': lambda a: a.bitcast(dtypes.float64).cast(dtypes.float32),
  'i32_to_f64': lambda a: a.cast(dtypes.int).cast(dtypes.float64),
  'u32_to_f64': lambda a: a.cast(dtypes.uint32).cast(dtypes.float64),
  'f16_to_i16': lambda a: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a),)).cast(dtypes.int16),
  'f16_to_u16': lambda a: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a),)).cast(dtypes.uint16),
  'i16_to_f16': lambda a: a.cast(dtypes.int16).cast(dtypes.half),
  'u16_to_f16': lambda a: a.cast(dtypes.uint16).cast(dtypes.half),
  'bf16_to_f32': lambda a: (((a.cast(dtypes.uint32) if a.dtype != dtypes.uint32 else a) & _u32(0xFFFF)) << _u32(16)).bitcast(dtypes.float32),
  'isNAN': _isnan, 'isSignalNAN': lambda a: _check_nan(a, False),
  'isQuietNAN': lambda a: _check_nan(a, True), 'cvtToQuietNAN': _cvt_quiet,
  'isDENORM': _is_denorm, 'exponent': _exponent, 'divWouldBeDenorm': _div_would_be_denorm, 'sign': _sign,
  'signext_from_bit': _signext_from_bit, 'ldexp': _ldexp, 'frexp_mant': _frexp_mant, 'mantissa': _frexp_mant,
  'frexp_exp': _frexp_exp, 'trig_preop_result': _trig_preop,
  's_ff1_i32_b32': lambda a: _ff1(a, 32), 's_ff1_i32_b64': lambda a: _ff1(a, 64),
  # Normalization conversions: map [-1,1] or [0,1] to integer range
  # Use floor(x + 0.5) for round-to-nearest
  # SNORM: round(value * 32767), range is [-32767, 32767] (hardware behavior)
  'f16_to_snorm': lambda a: _floor(_f16_extract(a).cast(dtypes.float32) * _const(dtypes.float32, 32767) + _const(dtypes.float32, 0.5)).cast(dtypes.int).cast(dtypes.int16),
  'f16_to_unorm': lambda a: _floor(_f16_extract(a).cast(dtypes.float32) * _const(dtypes.float32, 65535) + _const(dtypes.float32, 0.5)).cast(dtypes.uint16),
  'f32_to_snorm': lambda a: _floor(a.bitcast(dtypes.float32) * _const(dtypes.float32, 32767) + _const(dtypes.float32, 0.5)).cast(dtypes.int).cast(dtypes.int16),
  'f32_to_unorm': lambda a: _floor(a.bitcast(dtypes.float32) * _const(dtypes.float32, 65535) + _const(dtypes.float32, 0.5)).cast(dtypes.uint16),
  'f32_to_u8': lambda a: _f_to_u(a.bitcast(dtypes.float32), dtypes.uint8),
  # Integer truncation conversions
  'i32_to_i16': lambda a: a.cast(dtypes.int).cast(dtypes.int16),
  'u32_to_u16': lambda a: a.cast(dtypes.uint32).cast(dtypes.uint16),
  'u16_to_u32': lambda a: (a.cast(dtypes.uint32) & _u32(0xFFFF)),
  'u8_to_u32': lambda a: (a.cast(dtypes.uint32) & _u32(0xFF)),
  'u4_to_u32': lambda a: (a.cast(dtypes.uint32) & _u32(0xF)),
  # Signed extraction with sign extension for dot products
  'i16_to_i32': lambda a: _signext(a.cast(dtypes.uint32) & _u32(0xFFFF)),
  'i8_to_i32': lambda a: _signext(a.cast(dtypes.uint32) & _u32(0xFF)),
  'i4_to_i32': lambda a: _signext_4bit(a.cast(dtypes.uint32) & _u32(0xF)),
  # Float to int16 conversions
  'v_cvt_i16_f32': lambda a: UOp(Ops.TRUNC, dtypes.float32, (a.bitcast(dtypes.float32),)).cast(dtypes.int16),
  'v_cvt_u16_f32': lambda a: _f_to_u(a.bitcast(dtypes.float32), dtypes.uint16),
  # SAD (Sum of Absolute Differences) - sum |a_i - b_i| for 4 bytes + accumulator
  'v_sad_u8': lambda a, b, c: _sad_u8(a, b, c),
  'v_msad_u8': lambda a, b, c: _sad_u8(a, b, c, masked=True),
  # System NOPs - these are scheduling hints, no effect on emulation
  'MIN': lambda a, b: (a < b).where(a, b),
  's_nop': lambda a: _u32(0),
  # Address calculation for memory operations
  'CalcDsAddr': lambda a, o, *r: a.cast(dtypes.uint32) + o.cast(dtypes.uint32),
  'CalcGlobalAddr': lambda v, s, *r: v.cast(dtypes.uint64) + s.cast(dtypes.uint64),
}
for is_max, name in [(False, 'min'), (True, 'max')]:
  for dt, sfx in [(dtypes.float32, 'f32'), (dtypes.int, 'i32'), (dtypes.uint32, 'u32'), (dtypes.int16, 'i16'), (dtypes.uint16, 'u16')]:
    _FUNCS[f'v_{name}_{sfx}'] = lambda *a, im=is_max, d=dt: _minmax_reduce(im, d, *a)
    _FUNCS[f'v_{name}3_{sfx}'] = lambda *a, im=is_max, d=dt: _minmax_reduce(im, d, *a)
# f16 min/max/min3/max3/med3
for is_max, name in [(False, 'min'), (True, 'max')]:
  _FUNCS[f'v_{name}_f16'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.half, *[_f16_extract(x) for x in a])
  _FUNCS[f'v_{name}3_f16'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.half, *[_f16_extract(x) for x in a])
  _FUNCS[f'v_{name}_num_f16'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.half, *[_f16_extract(x) for x in a])
  _FUNCS[f'v_{name}_num_f32'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.float32, *a)
  _FUNCS[f'v_{name}3_num_f16'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.half, *[_f16_extract(x) for x in a])
  _FUNCS[f'v_{name}3_num_f32'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.float32, *a)
  _FUNCS[f'v_{name}imum_f16'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.half, *[_f16_extract(x) for x in a])
  _FUNCS[f'v_{name}imum_f32'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.float32, *a)
  _FUNCS[f'v_{name}imum3_f16'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.half, *[_f16_extract(x) for x in a])
  _FUNCS[f'v_{name}imum3_f32'] = lambda *a, im=is_max: _minmax_reduce(im, dtypes.float32, *a)

# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER/PARSER
# ═══════════════════════════════════════════════════════════════════════════════

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32, 'u64': dtypes.uint64, 'i64': dtypes.int64,
          'f64': dtypes.float64, 'b64': dtypes.uint64, 'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u8': dtypes.uint8, 'i8': dtypes.int8, 'b8': dtypes.uint8, 'u4': dtypes.uint8, 'i4': dtypes.int8, 'u1': dtypes.uint32}
_BITS_DT = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}
_NUM_SUFFIXES = ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f')
def _strip_suffix(num: str) -> tuple[str, str]:
  for sfx in _NUM_SUFFIXES:
    if num.endswith(sfx): return sfx, num[:-len(sfx)]
  return '', num
_SINGLE_CHAR = {'(': 'LPAREN', ')': 'RPAREN', '[': 'LBRACKET', ']': 'RBRACKET', '{': 'LBRACE', '}': 'RBRACE',
                ':': 'COLON', ',': 'COMMA', '?': 'QUESTION', '.': 'DOT', '=': 'EQUALS', "'": 'QUOTE'}

class Token:
  __slots__ = ('type', 'val')
  def __init__(self, type: str, val: str): self.type, self.val = type, val
  def __repr__(self): return f'{self.type}:{self.val}'

def tokenize(s: str) -> list[Token]:
  tokens, i, n = [], 0, len(s)
  while i < n:
    c = s[i]
    if c.isspace(): i += 1; continue
    if i + 1 < n and s[i:i+2] in ('+=', '-='):
      tokens.append(Token('ASSIGN_OP', s[i:i+2])); i += 2; continue
    if i + 1 < n and s[i:i+2] in ('||', '&&', '>=', '<=', '==', '!=', '<>', '>>', '<<', '**', '+:', '-:'):
      tokens.append(Token('OP', s[i:i+2])); i += 2; continue
    if c in '|^&><+-*/~!%': tokens.append(Token('OP', c)); i += 1; continue
    if (t := _SINGLE_CHAR.get(c)): tokens.append(Token(t, c)); i += 1; continue
    if c == ';': i += 1; continue
    if c.isdigit() or (c == '-' and i + 1 < n and s[i+1].isdigit()):
      start = i
      if c == '-': i += 1
      if i + 1 < n and s[i] == '0' and s[i+1] in 'xX':
        i += 2
        while i < n and s[i] in '0123456789abcdefABCDEF': i += 1
      else:
        while i < n and s[i].isdigit(): i += 1
        if i < n and s[i] == '.' and i + 1 < n and s[i+1].isdigit():
          i += 1
          while i < n and s[i].isdigit(): i += 1
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f'):
        if s[i:i+len(sfx)] == sfx: i += len(sfx); break
      tokens.append(Token('NUM', s[start:i])); continue
    if c.isalpha() or c == '_':
      start = i
      while i < n and (s[i].isalnum() or s[i] == '_'): i += 1
      tokens.append(Token('IDENT', s[start:i])); continue
    raise RuntimeError(f"unexpected char '{c}' at pos {i} in: {s}")
  tokens.append(Token('EOF', ''))
  return tokens

class Parser:
  def __init__(self, tokens: list[Token], vars: dict, funcs: dict | None = None):
    self.tokens, self.vars, self.funcs, self.pos = tokens, vars, funcs if funcs is not None else _FUNCS, 0

  def peek(self, offset=0) -> Token: return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]
  def at(self, *types) -> bool: return self.peek().type in types
  def _advance(self) -> Token: tok = self.tokens[self.pos]; self.pos += 1; return tok
  def eat(self, type: str) -> Token:
    if self.peek().type != type: raise RuntimeError(f"expected {type}, got {self.peek()}")
    return self._advance()
  def try_eat(self, type: str) -> Token | None: return self._advance() if self.peek().type == type else None
  def try_eat_val(self, val: str, type: str) -> Token | None:
    return self._advance() if self.peek().type == type and self.peek().val == val else None
  def eat_val(self, val: str, type: str) -> Token:
    if self.peek().type != type or self.peek().val != val: raise RuntimeError(f"expected {type}:{val}, got {self.peek()}")
    return self._advance()

  def parse(self) -> UOp:
    cond = self.binop(0)
    if self.try_eat('QUESTION'):
      then_val = self.parse()
      self.eat('COLON')
      return _to_bool(cond).where(then_val, self.parse())
    return cond

  def _apply_binop(self, left, right, op):
    if op in ('||', '&&', '|', '^', '&'): left, right = self._coerce_bitwise(left, right)
    elif op in ('>=', '<=', '>', '<', '==', '!=', '<>', '>>', '<<'): left, right = self._coerce_cmp(left, right)
    elif left.dtype != right.dtype: right = right.cast(left.dtype)
    match op:
      case '||' | '|': return left | right
      case '&&' | '&': return left & right
      case '^': return left ^ right
      case '==' | '<>': return left.eq(right) if op == '==' else left.ne(right)
      case '!=' : return left.ne(right)
      case '>=' | '<=' | '>' | '<': return self._cmp_nan(left, right, {'>=':(lambda a,b:a>=b),'<=':(lambda a,b:a<=b),'>':(lambda a,b:a>b),'<':(lambda a,b:a<b)}[op])
      case '>>' | '<<': return (left >> right) if op == '>>' else (left << right)
      case '+' | '-':
        if op == '-' and left.op == Ops.CONST and right.op == Ops.CONST: return _const(left.dtype, left.arg - right.arg)
        return (left + right) if op == '+' else (left - right)
      case '*' | '/': return (left * right) if op == '*' else (left / right)
      case '**': return UOp(Ops.EXP2, left.dtype, (right.cast(left.dtype),)) if left.op == Ops.CONST and left.arg == 2.0 else left

  _PREC = [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!=', '<>'), ('>=', '<=', '>', '<'), ('>>', '<<'), ('+', '-'), ('*', '/'), ('**',)]

  def binop(self, prec: int) -> UOp:
    if prec >= len(self._PREC): return self.unary()
    left = self.binop(prec + 1)
    ops = self._PREC[prec]
    while self.at('OP') and self.peek().val in ops:
      op = self.eat('OP').val
      left = self._apply_binop(left, self.binop(prec + 1), op)
    return left

  def unary(self) -> UOp:
    if self.try_eat_val('~', 'OP'):
      inner = self.unary()
      return inner ^ _const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)
    if self.try_eat_val('!', 'OP'):
      inner = self.unary()
      return inner.eq(_const(inner.dtype, 0))
    if self.try_eat_val('-', 'OP'):
      inner = self.unary()
      if inner.op == Ops.CONST:
        return _const(dtypes.int if inner.dtype == dtypes.uint32 else inner.dtype, -inner.arg)
      return inner.neg()
    if self.try_eat_val('+', 'OP'): return self.unary()
    return self.postfix()

  def postfix(self) -> UOp:
    base = self.primary()
    while True:
      if self.try_eat('DOT'):
        field = self.eat('IDENT').val
        base = self._handle_dot(base, field)
      elif self.at('LBRACKET'):
        base = self._handle_bracket(base)
      elif self.at('LBRACE'):
        base = self._handle_brace_index(base)
      else:
        break
    return base

  def primary(self) -> UOp:
    if self.try_eat('LPAREN'):
      e = self.parse()
      self.eat('RPAREN')
      return e
    if self.try_eat('LBRACE'):
      hi = self.parse()
      self.eat('COMMA')
      lo = self.parse()
      self.eat('RBRACE')
      return (hi.cast(dtypes.uint64) << _u64(32)) | lo.cast(dtypes.uint64)
    if self.at('NUM'):
      num = self.eat('NUM').val
      if self.try_eat('QUOTE'):
        return self._sized_literal(int(num.rstrip('ULlf')))
      return self._parse_number(num)
    if self.at('IDENT'):
      name = self.eat('IDENT').val
      if name == 'MEM':
        self.eat('LBRACKET')
        addr = self.parse()
        self.eat('RBRACKET')
        self.eat('DOT')
        dt_name = self.eat('IDENT').val
        return self._handle_mem_load(addr, DTYPES.get(dt_name, dtypes.uint32))
      if name == 'VGPR' and self.at('LBRACKET'):
        self.eat('LBRACKET')
        lane = self.parse()
        self.eat('RBRACKET')
        self.eat('LBRACKET')
        reg = self.parse()
        self.eat('RBRACKET')
        vgpr = self.vars.get('_vgpr')
        if vgpr is None: return _u32(0)
        return vgpr.index(_to_u32(reg) * _u32(32) + _to_u32(lane), ptr=True).load()
      if self.try_eat('LPAREN'):
        args = self._parse_args()
        self.eat('RPAREN')
        return self._call_func(name, args)
      if name == 'PI': return _const(dtypes.float32, 3.141592653589793)
      if name == 'INF': return _const(dtypes.float64, float('inf'))
      if name == 'NAN': return _const(dtypes.uint32, 0x7FC00000).bitcast(dtypes.float32)
      if name == 'UNDERFLOW_F32': return _const(dtypes.uint32, 1).bitcast(dtypes.float32)
      if name == 'OVERFLOW_F32': return _const(dtypes.uint32, 0x7F7FFFFF).bitcast(dtypes.float32)
      if name == 'UNDERFLOW_F64': return _const(dtypes.uint64, 1).bitcast(dtypes.float64)
      if name == 'OVERFLOW_F64': return _const(dtypes.uint64, 0x7FEFFFFFFFFFFFFF).bitcast(dtypes.float64)
      if name == 'WAVE32': return _const(dtypes.bool, True)
      if name == 'WAVE64': return _const(dtypes.bool, False)
      if name == 'WAVE_MODE' and self.try_eat('DOT') and self.try_eat_val('IEEE', 'IDENT'): return _u32(1)
      if self.try_eat('LBRACE'):
        idx = self.eat('NUM').val
        self.eat('RBRACE')
        # Handle VGPR{lane}[reg] - 2D array access after loop unrolling
        if name == 'VGPR' and self.at('LBRACKET'):
          self.eat('LBRACKET')
          reg = self.parse()
          self.eat('RBRACKET')
          vgpr = self.vars.get('_vgpr')
          if vgpr is None: return _u32(0)
          return vgpr.index(_to_u32(reg) * _u32(32) + _u32(int(idx)), ptr=True).load()
        elem = self.vars.get(f'{name}@{idx}', self.vars.get(f'{name}{idx}'))
        if elem is None:
          # Extract bit idx from base variable (like var[idx])
          base = self.vars.get(name)
          assert isinstance(base, UOp), f"unknown variable: {name}{idx}"
          dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
          elem = (base.cast(dt) >> _const(dt, int(idx))) & _const(dt, 1)
        if self.try_eat('DOT'):
          dt_name = self.eat('IDENT').val
          return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
        if self.at('LBRACKET'):
          return self._handle_bracket(elem, name + idx)
        return elem
      if self.at('LBRACKET') and name not in self.vars:
        self.eat('LBRACKET')
        first = self.parse()
        return self._handle_bracket_rest(first, _u32(0), name)
      if name in self.vars:
        v = self.vars[name]
        assert isinstance(v, UOp), f"expected UOp for {name}, got {type(v)}"
        return v
      raise RuntimeError(f"unknown variable: {name}")
    raise RuntimeError(f"unexpected token in primary: {self.peek()}")

  def _handle_dot(self, base, field: str) -> UOp:
    assert isinstance(base, UOp), f"expected UOp for dot access, got {type(base)}"
    if field == 'u64' and self.at('LBRACKET') and self.peek(1).type == 'IDENT' and self.peek(1).val == 'laneId':
      self.eat('LBRACKET')
      self.eat_val('laneId', 'IDENT')
      self.eat('RBRACKET')
      result = (base >> _to_u32(self.vars['laneId'])) & _u32(1)
      if self.try_eat('DOT'):
        dt_name = self.eat('IDENT').val
        return result.cast(DTYPES.get(dt_name, dtypes.uint32))
      return result
    dt = DTYPES.get(field)
    if dt is None: return base
    if dt == base.dtype: return base
    if dt.itemsize == 2 and base.dtype.itemsize == 4:
      return (base & _const(base.dtype, 0xFFFF)).cast(dtypes.uint16) if dt == dtypes.uint16 else (base & _const(base.dtype, 0xFFFF)).cast(dtypes.uint16).bitcast(dt)
    if field == 'i4': return _signext_4bit(base)
    return _cast_to(base, dt)

  def _handle_bracket(self, base, var_name: str | None = None) -> UOp:
    self.eat('LBRACKET')
    return self._handle_bracket_rest(self.parse(), base, var_name)

  def _handle_bracket_rest(self, first: UOp, base: UOp, var_name: str | None = None) -> UOp:
    if self.at('OP') and self.peek().val in ('+:', '-:'):
      op = self.eat('OP').val
      width = self.parse()
      self.eat('RBRACKET')
      if width.op == Ops.CONST:
        w = int(width.arg)
        return (base >> _to_u32(first)) & _const(base.dtype, (1 << w) - 1)
      return base
    if self.try_eat('COLON'):
      second = self.parse()
      self.eat('RBRACKET')
      if first.op == Ops.CONST and second.op == Ops.CONST:
        a, b = int(first.arg), int(second.arg)
        if a < b: return _bitreverse(base, b - a + 1)
        hi, lo = a, b
        if lo >= base.dtype.itemsize * 8:
          vn = var_name or self._find_var_name(base)
          if vn and f'{vn}{lo // 32}' in self.vars:
            base = self.vars[f'{vn}{lo // 32}']
            lo, hi = lo % 32, (hi % 32) + (lo % 32)
        return _extract_bits(base, hi, lo)
      # Dynamic bit slice: (base >> lo) & ((1 << (hi - lo + 1)) - 1)
      dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
      hi, lo = first.cast(dt), second.cast(dt)
      width = hi - lo + _const(dt, 1)
      mask = (_const(dt, 1) << width) - _const(dt, 1)
      return (base.cast(dt) >> lo) & mask
    self.eat('RBRACKET')
    dt_suffix = None
    if self.try_eat('DOT'):
      dt_suffix = DTYPES.get(self.eat('IDENT').val, dtypes.uint32)
    if var_name is None:
      var_name = self._find_var_name(base)
    if first.op == Ops.CONST:
      idx = int(first.arg)
      # Check for array element (var@idx)
      if var_name and f'{var_name}@{idx}' in self.vars:
        v = self.vars[f'{var_name}@{idx}']
        return _cast_to(v, dt_suffix) if dt_suffix else v
      # Bit extraction
      dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
      base_cast = base.cast(dt) if base.dtype != dt else base
      result = ((base_cast >> _const(dt, idx)) & _const(dt, 1))
      return _cast_to(result, dt_suffix) if dt_suffix else result
    if var_name:
      idx_u32 = _to_u32(first)
      elems = [(i, self.vars[f'{var_name}@{i}']) for i in range(256) if f'{var_name}@{i}' in self.vars]
      if elems:
        result = elems[-1][1]
        for ei, ev in reversed(elems[:-1]):
          if ev.dtype != result.dtype and ev.dtype.itemsize == result.dtype.itemsize: result = result.cast(ev.dtype)
          elif ev.dtype != result.dtype: ev = ev.cast(result.dtype)
          result = idx_u32.eq(_u32(ei)).where(ev, result)
        return result
    dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
    base_cast = base.cast(dt) if base.dtype != dt else base
    result = (base_cast >> first.cast(dt)) & _const(dt, 1)
    return _cast_to(result, dt_suffix) if dt_suffix else result

  def _handle_brace_index(self, base) -> UOp:
    self.eat('LBRACE')
    idx = self.eat('NUM').val
    self.eat('RBRACE')
    var_name = self._find_var_name(base)
    if var_name:
      elem = self.vars.get(f'{var_name}@{idx}', _u32(0))  # use @ to avoid collision with temps like A4
      if self.try_eat('DOT'):
        dt_name = self.eat('IDENT').val
        return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
      if self.at('LBRACKET'):
        return self._handle_bracket(elem)
      return elem
    return _u32(0)

  def _find_var_name(self, base: UOp) -> str | None:
    if base.op == Ops.DEFINE_VAR and base.arg: return base.arg[0]
    for name, v in self.vars.items():
      if isinstance(v, UOp) and v is base: return name
    return None

  def _sized_literal(self, bits: int) -> UOp:
    if self.at('IDENT') and self.peek().val in ('U', 'I', 'F', 'B'):
      type_char = self.eat('IDENT').val
      self.eat('LPAREN')
      inner = self.parse()
      self.eat('RPAREN')
      dt = {('U',32): dtypes.uint32, ('U',64): dtypes.uint64, ('I',32): dtypes.int, ('I',64): dtypes.int64,
            ('F',16): dtypes.half, ('F',32): dtypes.float32, ('F',64): dtypes.float64, ('B',32): dtypes.uint32, ('B',64): dtypes.uint64}.get((type_char, bits), dtypes.uint64 if bits > 32 else dtypes.uint32)
      if type_char == 'F' and inner.dtype in (dtypes.uint32, dtypes.uint64, dtypes.ulong, dtypes.int, dtypes.int64):
        if inner.dtype.itemsize != dt.itemsize: inner = inner.cast(dtypes.uint32 if dt.itemsize == 4 else dtypes.uint64)
        return inner.bitcast(dt)
      return inner.cast(dt)
    if self.at('IDENT'):
      ident = self.peek().val
      fmt = ident[0].lower()
      if fmt in ('h', 'b', 'd'):
        self.eat('IDENT')
        if len(ident) > 1: num = ident[1:]
        elif self.at('NUM'): num = self.eat('NUM').val
        elif self.at('IDENT'): num = self.eat('IDENT').val
        else: raise RuntimeError(f"expected number after {bits}'{fmt}")
        if fmt == 'h': val = int(num, 16)
        elif fmt == 'b': val = int(num, 2)
        else: val = int(num)
        return _const(_BITS_DT.get(bits, dtypes.uint32), val)
    if self.at('NUM') and self.peek().val.startswith('0x'):
      num = self.eat('NUM').val
      return _const(_BITS_DT.get(bits, dtypes.uint32), int(num, 16))
    if self.at('NUM') or (self.at('OP') and self.peek().val == '-'):
      neg = self.try_eat_val('-', 'OP') is not None
      suffix, num = _strip_suffix(self.eat('NUM').val)
      if num.startswith('0x'):
        val = int(num, 16)
        if neg: val = -val
      elif '.' in num:
        fval = float(num)
        if neg: fval = -fval
        return _const({16: dtypes.half, 32: dtypes.float32, 64: dtypes.float64}.get(bits, dtypes.float32), fval)
      else:
        val = int(num)
        if neg: val = -val
      dt = {1: dtypes.uint32, 8: dtypes.uint8, 16: dtypes.int16 if 'U' not in suffix else dtypes.uint16,
            32: dtypes.int if 'U' not in suffix else dtypes.uint32, 64: dtypes.int64 if 'U' not in suffix else dtypes.uint64}.get(bits, dtypes.uint32)
      return _const(dt, val)
    raise RuntimeError(f"unexpected token after {bits}': {self.peek()}")

  def _parse_number(self, num: str) -> UOp:
    if num.startswith('0x') or num.startswith('0X'):
      is_u64 = num.upper().endswith('ULL') or num.upper().endswith('LL') or num.upper().endswith('UL')
      return _const(dtypes.uint64 if is_u64 else dtypes.uint32, int(num.rstrip('ULul'), 16))
    suffix, num_str = _strip_suffix(num)
    if '.' in num_str or suffix in ('F', 'f'):
      return _const(dtypes.float32 if suffix in ('F', 'f') else dtypes.float64, float(num_str))
    val = int(num_str)
    if 'ULL' in suffix or 'LL' in suffix or 'L' in suffix: return _const(dtypes.uint64, val)
    if 'U' in suffix: return _const(dtypes.uint32, val)
    return _const(dtypes.int if val < 0 else dtypes.uint32, val)

  def _parse_args(self) -> list[UOp]:
    if self.at('RPAREN'): return []
    args = [self.parse()]
    while self.try_eat('COMMA'):
      args.append(self.parse())
    return args

  def _call_func(self, name: str, args: list[UOp]) -> UOp:
    if name in self.vars and isinstance(self.vars[name], tuple) and self.vars[name][0] == 'lambda':
      _, params, body = self.vars[name]
      lv = {**self.vars, **{p: a for p, a in zip(params, args)}}
      if ';' in body or '\n' in body or 'return' in body.lower():
        lines = [l.strip() for l in body.replace(';', '\n').split('\n') if l.strip() and not l.strip().startswith('//')]
        _, _, result = parse_block(lines, 0, lv, self.funcs)
        assert result is not None, f"lambda {name} must return a value"
        return result
      return parse_expr(body, lv, self.funcs)
    if name in self.funcs:
      return self.funcs[name](*args)
    raise RuntimeError(f"unknown function: {name}")

  def _handle_mem_load(self, addr: UOp, dt) -> UOp:
    mem = self.vars.get('_vmem') if '_vmem' in self.vars else self.vars.get('_lds')
    assert mem is not None, "memory load requires _vmem or _lds"
    adt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
    active = self.vars.get('_active')
    gate = (active,) if active is not None else ()
    byte_mem = mem.dtype.base == dtypes.uint8
    if byte_mem:
      idx = addr.cast(dtypes.int)
      if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
        val = _u32(0).cast(dtypes.uint64)
        for i in range(8): val = val | (mem.index(idx + _const(dtypes.int, i), *gate, ptr=True).load().cast(dtypes.uint64) << _u64(i * 8))
      elif dt in (dtypes.uint8, dtypes.int8):
        val = mem.index(idx, *gate, ptr=True).load().cast(dt)
      elif dt in (dtypes.uint16, dtypes.int16, dtypes.short):
        val = (mem.index(idx, *gate, ptr=True).load().cast(dtypes.uint32) | (mem.index(idx + _const(dtypes.int, 1), *gate, ptr=True).load().cast(dtypes.uint32) << _u32(8))).cast(dt)
      else:
        val = _u32(0)
        for i in range(4): val = val | (mem.index(idx + _const(dtypes.int, i), *gate, ptr=True).load().cast(dtypes.uint32) << _u32(i * 8))
    else:
      idx = (addr >> _const(addr.dtype, 2)).cast(dtypes.int)
      val = mem.index(idx, *gate)
      if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
        idx2 = ((addr + _const(adt, 4)) >> _const(adt, 2)).cast(dtypes.int)
        val = val.cast(dtypes.uint64) | (mem.index(idx2, *gate).cast(dtypes.uint64) << _u64(32))
      elif dt in (dtypes.uint8, dtypes.int8): val = (val >> ((addr & _const(adt, 3)).cast(dtypes.uint32) * _u32(8))) & _u32(0xFF)
      elif dt in (dtypes.uint16, dtypes.int16): val = (val >> (((addr >> _const(adt, 1)) & _const(adt, 1)).cast(dtypes.uint32) * _u32(16))) & _u32(0xFFFF)
    return val

  def _coerce_cmp(self, l: UOp, r: UOp) -> tuple[UOp, UOp]:
    if l.dtype != r.dtype:
      if r.dtype == dtypes.int and r.op == Ops.CONST and r.arg < 0: l = l.cast(dtypes.int)
      else: r = r.cast(l.dtype)
    return l, r

  def _coerce_bitwise(self, l: UOp, r: UOp) -> tuple[UOp, UOp]:
    if l.dtype != r.dtype:
      if l.dtype.itemsize == r.dtype.itemsize:
        t = dtypes.uint32 if l.dtype.itemsize == 4 else dtypes.uint64 if l.dtype.itemsize == 8 else l.dtype
        l, r = l.bitcast(t), r.bitcast(t)
      else: r = r.cast(l.dtype)
    return l, r

  def _cmp_nan(self, l: UOp, r: UOp, fn) -> UOp:
    result = fn(l, r)
    if l.dtype in (dtypes.float32, dtypes.float64, dtypes.half):
      return result & _isnan(l).logical_not() & _isnan(r).logical_not()
    return result

def _match_bracket(toks: list[Token], start: int) -> tuple[int, list[Token]]:
  """Match brackets from start, return (end_idx, inner_tokens)."""
  j, depth = start + 1, 1
  while j < len(toks) and depth > 0:
    if toks[j].type == 'LBRACKET': depth += 1
    elif toks[j].type == 'RBRACKET': depth -= 1
    j += 1
  return j, [t for t in toks[start+1:j-1] if t.type != 'EOF']

def _tok_str(toks: list[Token]) -> str: return ' '.join(t.val for t in toks if t.type != 'EOF')
def parse_tokens(toks: list[Token], vars: dict[str, VarVal], funcs: dict | None = None) -> UOp:
  return Parser(toks, vars, funcs).parse()

# Unified block parser for pcode
def _subst_loop_var(line: str, loop_var: str, val: int) -> str:
  """Substitute loop variable with its value."""
  toks = tokenize(line)
  return ' '.join(str(val) if t.type == 'IDENT' and t.val == loop_var else t.val for t in toks if t.type != 'EOF')

def _set_bits(old: UOp, val: UOp, width: int, offset: int) -> UOp:
  """Set bits [offset:offset+width) in old to val, masking and shifting appropriately."""
  mask = _u32(((1 << width) - 1) << offset)
  v = (val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val) & _u32((1 << width) - 1)
  return (old & (mask ^ _u32(0xFFFFFFFF))) | (v << _u32(offset))

def _find_paren_end(s: str, start: int = 0, open_ch: str = '(', close_ch: str = ')') -> int:
  """Find index of matching close paren, starting after the open paren at start."""
  depth = 0
  for j, ch in enumerate(s[start:], start):
    if ch == open_ch: depth += 1
    elif ch == close_ch:
      depth -= 1
      if depth == 0: return j
  return len(s)

def parse_block(lines: list[str], start: int, vars: dict[str, VarVal], funcs: dict | None = None,
                assigns: list | None = None) -> tuple[int, dict[str, VarVal], UOp | None]:
  """Parse a block of pcode. Returns (next_line, block_assigns, return_value).
  If assigns list is provided, side effects (MEM/VGPR writes) are appended to it."""
  if funcs is None: funcs = _FUNCS
  block_assigns: dict[str, VarVal] = {}
  i = start

  while i < len(lines):
    line = lines[i]
    toks = tokenize(line)
    if toks[0].type != 'IDENT' and toks[0].type != 'LBRACE': i += 1; continue
    first = toks[0].val.lower() if toks[0].type == 'IDENT' else '{'

    # Block terminators
    if first in ('elsif', 'else', 'endif', 'endfor'): break

    # return expr (lambda bodies)
    if first == 'return':
      rest = line[line.lower().find('return') + 6:].strip()
      return i + 1, block_assigns, parse_expr(rest, vars, funcs)

    # for loop
    if first == 'for':
      # Parse: for VAR in [SIZE']START : [SIZE']END do
      p = Parser(toks, vars, funcs)
      p.eat_val('for', 'IDENT')
      loop_var = p.eat('IDENT').val
      p.eat_val('in', 'IDENT')
      def parse_bound():
        if p.at('NUM') and p.peek(1).type == 'QUOTE': p.eat('NUM'); p.eat('QUOTE')
        if p.at('NUM'): return int(p.eat('NUM').val.rstrip('UuLl'))
        expr = p.parse().simplify()
        assert expr.op == Ops.CONST, f"loop bound must be constant, got {expr}"
        return int(expr.arg)
      start_val = parse_bound()
      p.eat('COLON')
      end_val = parse_bound()
      # Collect body
      i += 1
      body_lines: list[str] = []
      depth = 1
      while i < len(lines) and depth > 0:
        btoks = tokenize(lines[i])
        if btoks[0].type == 'IDENT':
          if btoks[0].val.lower() == 'for': depth += 1
          elif btoks[0].val.lower() == 'endfor': depth -= 1
        if depth > 0: body_lines.append(lines[i])
        i += 1
      # Execute loop with break support
      has_break = any('break' in bl.lower() for bl in body_lines)
      found_var = f'_found_{id(body_lines)}' if has_break else None
      if found_var: vars[found_var] = block_assigns[found_var] = _const(dtypes.bool, False)
      for loop_i in range(start_val, end_val + 1):
        subst_lines = [_subst_loop_var(bl, loop_var, loop_i) for bl in body_lines if not (has_break and bl.strip().lower() == 'break')]
        _, iter_assigns, _ = parse_block(subst_lines, 0, {**vars, **block_assigns}, funcs, assigns)
        if has_break:
          assert found_var is not None
          found = block_assigns.get(found_var, vars.get(found_var))
          assert isinstance(found, UOp)
          not_found = found.eq(_const(dtypes.bool, False))
          for var, val in iter_assigns.items():
            if var != found_var and isinstance(val, UOp):
              old = block_assigns.get(var, vars.get(var, _u32(0)))
              if isinstance(old, UOp):
                block_assigns[var] = vars[var] = not_found.where(val, old.cast(val.dtype) if val.dtype != old.dtype and val.dtype.itemsize == old.dtype.itemsize else old)
          for j, bl in enumerate(body_lines):
            bl_l = bl.strip().lower()
            if bl_l.startswith('if ') and bl_l.endswith(' then'):
              if any(body_lines[k].strip().lower() == 'break' for k in range(j+1, len(body_lines))):
                cond_str = _subst_loop_var(bl.strip()[3:-5].strip(), loop_var, loop_i)
                cond = _to_bool(parse_expr(cond_str, vars, funcs))
                block_assigns[found_var] = vars[found_var] = not_found.where(cond, found)
                break
        else:
          block_assigns.update(iter_assigns); vars.update(iter_assigns)
      continue

    # declare
    if first == 'declare':
      # Initialize scalar declarations (skip arrays and vars already passed as srcs)
      if '[' not in line and len(toks) >= 2 and toks[1].type == 'IDENT':
        vars.setdefault(toks[1].val, _u32(0))
      i += 1; continue

    # lambda definition
    if first != '{' and '=' in line and 'lambda' in line and any(t.type == 'IDENT' and t.val == 'lambda' for t in toks):
      name = toks[0].val
      body_start = line[line.find('(', line.find('lambda')):]
      params_end = _find_paren_end(body_start) + 1
      params = [p.strip() for p in body_start[1:params_end-1].split(',') if p.strip()]
      rest = body_start[params_end:].strip()
      if rest.startswith('('):
        body_end = _find_paren_end(rest)
        if body_end < len(rest):  # found matching paren on same line
          body = rest[1:body_end].strip()
          i += 1
        else:  # multiline body
          body_lines_lst, depth = [rest[1:]], 1
          i += 1
          while i < len(lines) and depth > 0:
            for j, ch in enumerate(lines[i]):
              if ch == '(': depth += 1
              elif ch == ')':
                depth -= 1
                if depth == 0: body_lines_lst.append(lines[i][:j]); break
            else: body_lines_lst.append(lines[i])
            i += 1
          body = '\n'.join(body_lines_lst).strip()
        vars[name] = ('lambda', params, body)
        continue

    # MEM assignment: MEM[addr].type (+|-)?= value
    if first == 'mem' and toks[1].type == 'LBRACKET':
      j, addr_toks = _match_bracket(toks, 1)
      addr = parse_tokens(addr_toks, vars, funcs)
      if j < len(toks) and toks[j].type == 'DOT': j += 1
      dt_name = toks[j].val if j < len(toks) and toks[j].type == 'IDENT' else 'u32'
      dt, j = DTYPES.get(dt_name, dtypes.uint32), j + 1
      compound_op = None
      if j < len(toks) and toks[j].type == 'ASSIGN_OP': compound_op = toks[j].val; j += 1
      elif j < len(toks) and toks[j].type == 'EQUALS': j += 1
      rhs = parse_tokens(toks[j:], vars, funcs)
      if compound_op:
        mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')
        if isinstance(mem, UOp):
          adt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
          idx = (addr >> _const(adt, 2)).cast(dtypes.int)
          old = mem.index(idx)
          if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
            old = old.cast(dtypes.uint64) | (mem.index(((addr + _const(adt, 4)) >> _const(adt, 2)).cast(dtypes.int)).cast(dtypes.uint64) << _u64(32))
          rhs = (old + rhs) if compound_op == '+=' else (old - rhs)
      if assigns is not None: assigns.append((f'MEM[{_tok_str(addr_toks)}].{dt_name}', (addr, rhs)))
      i += 1; continue

    # VGPR assignment: VGPR[lane][reg] = value
    if first == 'vgpr' and toks[1].type == 'LBRACKET':
      j, lane_toks = _match_bracket(toks, 1)
      if j < len(toks) and toks[j].type == 'LBRACKET':
        j, reg_toks = _match_bracket(toks, j)
        if j < len(toks) and toks[j].type == 'DOT': j += 2  # skip .type suffix
        if j < len(toks) and toks[j].type == 'EQUALS': j += 1
        ln, rg, val = parse_tokens(lane_toks, vars, funcs), parse_tokens(reg_toks, vars, funcs), parse_tokens(toks[j:], vars, funcs)
        if assigns is not None: assigns.append((f'VGPR[{_tok_str(lane_toks)}][{_tok_str(reg_toks)}]', (_to_u32(rg) * _u32(32) + _to_u32(ln), val)))
        i += 1; continue

    # Compound destination: {hi.type, lo.type} = value
    if first == '{':
      j = 1
      if j+2 < len(toks) and toks[j].type == 'IDENT' and toks[j+1].type == 'DOT':
        hi_var, hi_type = toks[j].val, toks[j+2].val
        j += 3
        if j < len(toks) and toks[j].type == 'COMMA': j += 1
        if j+2 < len(toks) and toks[j].type == 'IDENT' and toks[j+1].type == 'DOT':
          lo_var, lo_type = toks[j].val, toks[j+2].val
          j += 3
          if j < len(toks) and toks[j].type == 'RBRACE': j += 1
          if j < len(toks) and toks[j].type == 'EQUALS': j += 1
          val = parse_tokens(toks[j:], vars, funcs)
          lo_dt, hi_dt = DTYPES.get(lo_type, dtypes.uint64), DTYPES.get(hi_type, dtypes.uint32)
          lo_bits = 64 if lo_dt in (dtypes.uint64, dtypes.int64) else 32
          lo_val = val.cast(lo_dt) if val.dtype.itemsize * 8 <= lo_bits else (val & _const(val.dtype, (1 << lo_bits) - 1)).cast(lo_dt)
          hi_val = (val >> _const(val.dtype, lo_bits)).cast(hi_dt)
          block_assigns[lo_var] = vars[lo_var] = lo_val
          block_assigns[hi_var] = vars[hi_var] = hi_val
          if assigns is not None: assigns.extend([(f'{lo_var}.{lo_type}', lo_val), (f'{hi_var}.{hi_type}', hi_val)])
          i += 1; continue

    # Bit slice/index: var[hi:lo] = value, var.type[hi:lo] = value, or var[expr] = value
    if len(toks) >= 5 and toks[0].type == 'IDENT' and (toks[1].type == 'LBRACKET' or (toks[1].type == 'DOT' and toks[3].type == 'LBRACKET')):
      bracket_start = 2 if toks[1].type == 'LBRACKET' else 4
      j = bracket_start
      colon_pos = None
      while j < len(toks) and toks[j].type != 'RBRACKET':
        if toks[j].type == 'COLON': colon_pos = j
        j += 1
      var = toks[0].val
      if colon_pos is not None:  # bit slice: var[hi:lo]
        hi_str = ' '.join(t.val for t in toks[bracket_start:colon_pos] if t.type != 'EOF')
        lo_str = ' '.join(t.val for t in toks[colon_pos+1:j] if t.type != 'EOF')
        try:
          hi_val, lo_val = int(eval(hi_str)), int(eval(lo_str))
          hi, lo = max(hi_val, lo_val), min(hi_val, lo_val)
          j += 1
          if j < len(toks) and toks[j].type == 'DOT': j += 2
          if j < len(toks) and toks[j].type == 'EQUALS': j += 1
          val = parse_tokens(toks[j:], vars, funcs)
          dt_suffix = toks[2].val if toks[1].type == 'DOT' else None
          if assigns is not None: assigns.append((f'{var}[{hi}:{lo}]' + (f'.{dt_suffix}' if dt_suffix else ''), val))
          if var not in vars: vars[var] = _const(dtypes.uint64 if hi >= 32 else dtypes.uint32, 0)
          old = block_assigns.get(var, vars.get(var))
          block_assigns[var] = vars[var] = _set_bits(old, _val_to_bits(val), hi - lo + 1, lo)
          i += 1; continue
        except: pass
      elif toks[1].type == 'LBRACKET':  # bit index: var[expr] (only for var[...], not var.type[...])
        existing = block_assigns.get(var, vars.get(var))
        if existing is not None and isinstance(existing, UOp) and not any(f'{var}{k}' in vars or f'{var}{k}' in block_assigns for k in range(8)):
          bit_toks = toks[2:j]
          j += 1
          while j < len(toks) and toks[j].type != 'EQUALS': j += 1
          if j < len(toks):
            block_assigns[var] = vars[var] = _set_bit(existing, _to_u32(parse_tokens(bit_toks, vars, funcs)), parse_tokens(toks[j+1:], vars, funcs))
            i += 1; continue

    # Array element: var[idx] = value (static index) or var[expr] = value (dynamic)
    if len(toks) >= 4 and toks[0].type == 'IDENT' and toks[1].type == 'LBRACKET':
      var = toks[0].val
      j, idx_toks = _match_bracket(toks, 1)
      if j < len(toks) and toks[j].type == 'EQUALS':
        # Static index: var[NUM] = value
        if len(idx_toks) == 1 and idx_toks[0].type == 'NUM':
          idx = int(idx_toks[0].val.rstrip('UuLl'))
          val = parse_tokens(toks[j+1:], vars, funcs)
          existing = block_assigns.get(var, vars.get(var))
          if existing is not None and isinstance(existing, UOp):
            block_assigns[var] = vars[var] = _set_bit(existing, _u32(idx), val)
          else:
            block_assigns[f'{var}@{idx}'] = vars[f'{var}@{idx}'] = val
          i += 1; continue
        # Dynamic index: var[expr] = value where var has @-elements
        elems = [(k.split('@')[1], v) for k, v in {**vars, **block_assigns}.items() if k.startswith(f'{var}@') and isinstance(v, UOp)]
        if elems:
          idx_expr = parse_tokens(idx_toks, vars, funcs)
          val = parse_tokens(toks[j+1:], vars, funcs)
          for elem_idx_str, old_elem in elems:
            elem_idx = int(elem_idx_str)
            cond = _to_u32(idx_expr).eq(_u32(elem_idx))
            new_val = cond.where(val.cast(old_elem.dtype) if val.dtype != old_elem.dtype else val, old_elem)
            block_assigns[f'{var}@{elem_idx}'] = vars[f'{var}@{elem_idx}'] = new_val
          i += 1; continue

    # Compound assignment: var += or var -=
    assign_op = next((j for j, t in enumerate(toks) if t.type == 'ASSIGN_OP'), None)
    if assign_op is not None:
      var = toks[0].val
      old = block_assigns.get(var, vars.get(var, _u32(0)))
      rhs = parse_tokens(toks[assign_op+1:], vars, funcs)
      if rhs.dtype != old.dtype: rhs = rhs.cast(old.dtype)
      block_assigns[var] = vars[var] = (old + rhs) if toks[assign_op].val == '+=' else (old - rhs)
      i += 1; continue

    # Typed element: var.type[idx] = value
    if len(toks) >= 7 and toks[0].type == 'IDENT' and toks[1].type == 'DOT' and toks[2].type == 'IDENT' and toks[3].type == 'LBRACKET' and toks[4].type == 'NUM':
      var, dt_name, idx = toks[0].val, toks[2].val, int(toks[4].val)
      dt = DTYPES.get(dt_name, dtypes.uint32)
      j = 6
      while j < len(toks) and toks[j].type != 'EQUALS': j += 1
      if j < len(toks):
        val, old = parse_tokens(toks[j+1:], vars, funcs), block_assigns.get(var, vars.get(var, _u32(0)))
        bw = dt.itemsize * 8
        block_assigns[var] = vars[var] = _set_bits(old, val, bw, idx * bw)
        if assigns is not None: assigns.append((f'{var}.{dt_name}[{idx}]', val))
        i += 1; continue

    # Dynamic bit: var.type[expr_with_brackets] = value
    if len(toks) >= 5 and toks[0].type == 'IDENT' and toks[1].type == 'DOT' and toks[2].type == 'IDENT' and toks[3].type == 'LBRACKET':
      j, depth, has_inner = 4, 1, False
      while j < len(toks) and depth > 0:
        if toks[j].type == 'LBRACKET': depth += 1; has_inner = True
        elif toks[j].type == 'RBRACKET': depth -= 1
        j += 1
      if has_inner:
        var = toks[0].val
        bit_pos = _to_u32(parse_tokens(toks[4:j-1], vars, funcs))
        while j < len(toks) and toks[j].type != 'EQUALS': j += 1
        if j < len(toks):
          val = parse_tokens(toks[j+1:], vars, funcs)
          old = block_assigns.get(var, vars.get(var, _u32(0)))
          block_assigns[var] = vars[var] = _set_bit(old, bit_pos, val)
          i += 1; continue

    # If/elsif/else - skip branches with statically false conditions (WAVE32/WAVE64)
    if first == 'if':
      def parse_cond(s, kw):
        ll = s.lower()
        return _to_bool(parse_expr(s[ll.find(kw) + len(kw):ll.rfind('then')].strip(), vars, funcs))
      def is_const(c, v): return c.op == Ops.CONST and c.arg is v
      cond = parse_cond(line, 'if')
      conditions: list[tuple[UOp, UOp | dict[str, VarVal] | None]] = [(cond, None)] if not is_const(cond, False) else []
      else_branch: tuple[UOp | None, dict[str, VarVal]] = (None, {})
      vars_snap = dict(vars)
      static_true = is_const(cond, True)  # track if any condition is statically true
      i += 1
      i, branch, ret = parse_block(lines, i, vars, funcs, assigns if not is_const(cond, False) else None)
      if conditions: conditions[0] = (cond, ret if ret is not None else branch)
      vars.clear(); vars.update(vars_snap)
      while i < len(lines):
        ltoks = tokenize(lines[i])
        if ltoks[0].type != 'IDENT': break
        lf = ltoks[0].val.lower()
        if lf == 'elsif':
          c = parse_cond(lines[i], 'elsif')
          take = not static_true and not is_const(c, False)
          i += 1; i, branch, ret = parse_block(lines, i, vars, funcs, assigns if take else None)
          if take:
            conditions.append((c, ret if ret is not None else branch))
            if is_const(c, True): static_true = True
          vars.clear(); vars.update(vars_snap)
        elif lf == 'else':
          i += 1
          i, branch, ret = parse_block(lines, i, vars, funcs, assigns if not static_true else None)
          if not static_true: else_branch = (ret, branch)
          vars.clear(); vars.update(vars_snap)
        elif lf == 'endif': i += 1; break
        else: break
      # Check if any branch returned a value (lambda-style)
      if any(isinstance(br, UOp) for _, br in conditions):
        result = else_branch[0]
        for c, rv in reversed(conditions):
          if isinstance(rv, UOp) and isinstance(result, UOp):
            if rv.dtype != result.dtype and rv.dtype.itemsize == result.dtype.itemsize: result = result.cast(rv.dtype)
            result = c.where(rv, result)
        return i, block_assigns, result
      # If statically true, use that branch directly; otherwise merge with WHERE
      if static_true:
        ba = next((b for c, b in conditions if is_const(c, True) and isinstance(b, dict)), {})
        block_assigns.update(ba); vars.update(ba)
      else:
        else_assigns = else_branch[1]
        all_vars = set().union(*[ba.keys() for _, ba in conditions if isinstance(ba, dict)], else_assigns.keys())
        for var in all_vars:
          res: Any = else_assigns.get(var, block_assigns.get(var, vars.get(var, _u32(0))))
          for cond, ba in reversed(conditions):
            if isinstance(ba, dict) and var in ba:
              tv = ba[var]
              if isinstance(tv, UOp) and isinstance(res, UOp):
                res = cond.where(tv, res.cast(tv.dtype) if tv.dtype != res.dtype and tv.dtype.itemsize == res.dtype.itemsize else res)
          block_assigns[var] = vars[var] = res
      continue

    # Regular assignment: var = value
    for j, t in enumerate(toks):
      if t.type == 'EQUALS':
        if any(toks[k].type == 'OP' and toks[k].val in ('<', '>', '!', '=') for k in range(j)): break
        base_var = toks[0].val
        block_assigns[base_var] = vars[base_var] = parse_tokens(toks[j+1:], vars, funcs)
        i += 1; break
    else: i += 1
  return i, block_assigns, None

def parse_expr(expr: str, vars: dict[str, VarVal], funcs: dict | None = None) -> UOp:
  return parse_tokens(tokenize(expr.strip().rstrip(';')), vars, funcs)

