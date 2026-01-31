# Tokenizer-based expression parser for AMD pcode
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32, 'u64': dtypes.uint64, 'i64': dtypes.int64,
          'f64': dtypes.float64, 'b64': dtypes.uint64, 'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u8': dtypes.uint8, 'i8': dtypes.int8, 'b8': dtypes.uint8, 'u1': dtypes.uint32}
_BITS_DT = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}

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

def _minmax_reduce(is_max, dt, args):
  def cast(v): return v.bitcast(dt) if dt == dtypes.float32 and v.dtype == dtypes.uint32 else v.cast(dt)
  def minmax(a, b):
    if dt in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64):
      return (a > b).where(a, b) if is_max else (a < b).where(a, b)
    return a.maximum(b) if is_max else a.minimum(b)
  result = cast(args[0])
  for a in args[1:]:
    b = cast(a)
    if dt == dtypes.float32: result = _isnan(result).where(b, _isnan(b).where(result, minmax(result, b)))
    else: result = minmax(result, b)
  return result

# Token types
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
    if c == '(': tokens.append(Token('LPAREN', c)); i += 1; continue
    if c == ')': tokens.append(Token('RPAREN', c)); i += 1; continue
    if c == '[': tokens.append(Token('LBRACKET', c)); i += 1; continue
    if c == ']': tokens.append(Token('RBRACKET', c)); i += 1; continue
    if c == '{': tokens.append(Token('LBRACE', c)); i += 1; continue
    if c == '}': tokens.append(Token('RBRACE', c)); i += 1; continue
    if c == ':': tokens.append(Token('COLON', c)); i += 1; continue
    if c == ',': tokens.append(Token('COMMA', c)); i += 1; continue
    if c == '?': tokens.append(Token('QUESTION', c)); i += 1; continue
    if c == '.': tokens.append(Token('DOT', c)); i += 1; continue
    if c == '=': tokens.append(Token('EQUALS', c)); i += 1; continue
    if c == "'": tokens.append(Token('QUOTE', c)); i += 1; continue
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
  def at_val(self, *vals) -> bool: return self.peek().val in vals
  def eat(self, type: str) -> Token:
    if self.peek().type != type: raise RuntimeError(f"expected {type}, got {self.peek()}")
    tok = self.tokens[self.pos]; self.pos += 1; return tok
  def try_eat(self, type: str) -> Token | None:
    if self.peek().type == type: return self.eat(type)
    return None
  def try_eat_val(self, val: str) -> Token | None:
    if self.peek().val == val: tok = self.tokens[self.pos]; self.pos += 1; return tok
    return None

  def parse(self) -> UOp: return self.ternary()
  def expr_top(self) -> UOp: return self.ternary()

  def ternary(self) -> UOp:
    cond = self.binop(0)
    if self.try_eat('QUESTION'):
      then_val, else_val = self.ternary(), (self.eat('COLON'), self.ternary())[1]
      return _to_bool(cond).where(then_val, else_val)
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
    if self.at('OP') and self.peek().val == '~':
      self.eat('OP'); inner = self.unary()
      return inner ^ _const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)
    if self.at('OP') and self.peek().val == '!':
      self.eat('OP'); inner = self.unary()
      return inner.eq(_const(inner.dtype, 0))
    if self.at('OP') and self.peek().val == '-':
      self.eat('OP'); inner = self.unary()
      if inner.op == Ops.CONST:
        return _const(dtypes.int if inner.dtype == dtypes.uint32 else inner.dtype, -inner.arg)
      return inner.neg()
    if self.at('OP') and self.peek().val == '+':
      self.eat('OP')
      return self.unary()
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
      e = self.expr_top()
      self.eat('RPAREN')
      return e
    if self.try_eat('LBRACE'):
      hi = self.expr_top()
      self.eat('COMMA')
      lo = self.expr_top()
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
        addr = self.expr_top()
        self.eat('RBRACKET')
        self.eat('DOT')
        dt_name = self.eat('IDENT').val
        return self._handle_mem_load(addr, DTYPES.get(dt_name, dtypes.uint32))
      if name == 'VGPR':
        self.eat('LBRACKET')
        lane = self.expr_top()
        self.eat('RBRACKET')
        self.eat('LBRACKET')
        reg = self.expr_top()
        self.eat('RBRACKET')
        vgpr = self.vars.get('_vgpr')
        if vgpr is None: return _u32(0)
        return vgpr.index((_to_u32(reg) * _u32(32) + _to_u32(lane)).cast(dtypes.index), ptr=True).load()
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
      if self.at('LBRACE'):
        self.eat('LBRACE')
        idx = self.eat('NUM').val
        self.eat('RBRACE')
        elem = self.vars.get(f'{name}{idx}', _u32(0))
        if self.try_eat('DOT'):
          dt_name = self.eat('IDENT').val
          return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
        if self.at('LBRACKET'):
          return self._handle_bracket_with_name(elem, name + idx)
        return elem
      if self.at('LBRACKET') and name not in self.vars:
        self.eat('LBRACKET')
        if self.at('NUM'):
          idx = int(self.peek().val)
          if f'{name}{idx}' in self.vars:
            self.eat('NUM')
            self.eat('RBRACKET')
            elem = self.vars[f'{name}{idx}']
            if self.try_eat('DOT'):
              dt_name = self.eat('IDENT').val
              return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
            return elem
        first = self.expr_top()
        return self._handle_bracket_rest(first, _u32(0), name)
      if name in self.vars:
        v = self.vars[name]
        return v if isinstance(v, UOp) else _u32(0) if isinstance(v, dict) else _u32(0)
      return _u32(0)
    raise RuntimeError(f"unexpected token in primary: {self.peek()}")

  def _handle_dot(self, base, field: str) -> UOp:
    if isinstance(base, str): return _u32(0)
    if not isinstance(base, UOp):
      if isinstance(base, dict): return base.get(field, _u32(0))
      return _u32(0)
    if field == 'u64' and self.at('LBRACKET') and self.peek(1).type == 'IDENT' and self.peek(1).val == 'laneId':
      self.eat('LBRACKET')
      self.eat('IDENT')
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
    return _cast_to(base, dt)

  def _handle_bracket(self, base, var_name: str | None = None) -> UOp:
    self.eat('LBRACKET')
    first = self.expr_top()
    return self._handle_bracket_rest(first, base, var_name)

  def _handle_bracket_with_name(self, base, var_name: str) -> UOp:
    self.eat('LBRACKET')
    first = self.expr_top()
    return self._handle_bracket_rest(first, base, var_name)

  def _handle_bracket_rest(self, first: UOp, base: UOp, var_name: str | None = None) -> UOp:
    if self.at('OP') and self.peek().val in ('+:', '-:'):
      op = self.eat('OP').val
      width = self.expr_top()
      self.eat('RBRACKET')
      if width.op == Ops.CONST:
        w = int(width.arg)
        return (base >> _to_u32(first)) & _const(base.dtype, (1 << w) - 1)
      return base
    if self.try_eat('COLON'):
      second = self.expr_top()
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
      if var_name and f'{var_name}{idx}' in self.vars:
        v = self.vars[f'{var_name}{idx}']
        return _cast_to(v, dt_suffix) if dt_suffix else v
      dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
      base_cast = base.cast(dt) if base.dtype != dt else base
      result = ((base_cast >> _const(dt, idx)) & _const(dt, 1))
      return _cast_to(result, dt_suffix) if dt_suffix else result
    if var_name:
      idx_u32 = _to_u32(first)
      elems = [(i, self.vars[f'{var_name}{i}']) for i in range(256) if f'{var_name}{i}' in self.vars]
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
      elem = self.vars.get(f'{var_name}{idx}', _u32(0))
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
      inner = self.expr_top()
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
      neg = self.try_eat_val('-') is not None
      num = self.eat('NUM').val
      suffix = ''
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f'):
        if num.endswith(sfx): suffix, num = sfx, num[:-len(sfx)]; break
      if num.startswith('0x'):
        val = int(num, 16)
        if neg: val = -val
      elif '.' in num:
        val = float(num)
        if neg: val = -val
        return _const({16: dtypes.half, 32: dtypes.float32, 64: dtypes.float64}.get(bits, dtypes.float32), val)
      else:
        val = int(num)
        if neg: val = -val
      dt = {1: dtypes.uint32, 8: dtypes.uint8, 16: dtypes.int16 if 'U' not in suffix else dtypes.uint16,
            32: dtypes.int if 'U' not in suffix else dtypes.uint32, 64: dtypes.int64 if 'U' not in suffix else dtypes.uint64}.get(bits, dtypes.uint32)
      return _const(dt, val)
    raise RuntimeError(f"unexpected token after {bits}': {self.peek()}")

  def _parse_number(self, num: str) -> UOp:
    suffix = ''
    if num.startswith('0x') or num.startswith('0X'):
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L'):
        if num.endswith(sfx): suffix, num = sfx, num[:-len(sfx)]; break
      return _const(dtypes.uint64, int(num, 16))
    for sfx in ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f'):
      if num.endswith(sfx): suffix, num = sfx, num[:-len(sfx)]; break
    if '.' in num or suffix in ('F', 'f'):
      return _const(dtypes.float32 if suffix in ('F', 'f') else dtypes.float64, float(num))
    val = int(num)
    if 'ULL' in suffix: return _const(dtypes.uint64, val)
    if 'LL' in suffix or 'L' in suffix: return _const(dtypes.uint64, val)
    if 'U' in suffix: return _const(dtypes.uint32, val)
    return _const(dtypes.int if val < 0 else dtypes.uint32, val)

  def _parse_args(self) -> list[UOp]:
    if self.at('RPAREN'): return []
    args = [self.expr_top()]
    while self.try_eat('COMMA'):
      args.append(self.expr_top())
    return args

  def _call_func(self, name: str, args: list[UOp]) -> UOp:
    if name in self.vars and isinstance(self.vars[name], tuple) and self.vars[name][0] == 'lambda':
      _, params, body = self.vars[name]
      lv = {**self.vars, **{p: a for p, a in zip(params, args)}}
      if ';' in body or '\n' in body or 'return' in body.lower():
        return _parse_lambda_body(body, lv, self.funcs)
      return parse_expr(body, lv, self.funcs)
    if name in self.funcs:
      return self.funcs[name](args)
    raise RuntimeError(f"unknown function: {name}")

  def _handle_mem_load(self, addr: UOp, dt) -> UOp:
    mem = self.vars.get('_vmem') if '_vmem' in self.vars else self.vars.get('_lds')
    if mem is None: return _const(dt, 0)
    adt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
    active = self.vars.get('_active')
    byte_mem = mem.dtype.base == dtypes.uint8
    if byte_mem:
      idx = addr.cast(dtypes.index)
      idx = idx.valid(active) if active is not None else idx
      if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
        val = _u32(0).cast(dtypes.uint64)
        for i in range(8): val = val | (mem.index(idx + _const(dtypes.index, i), ptr=True).load().cast(dtypes.uint64) << _u64(i * 8))
      elif dt in (dtypes.uint8, dtypes.int8):
        val = mem.index(idx, ptr=True).load().cast(dt)
      elif dt in (dtypes.uint16, dtypes.int16, dtypes.short):
        val = (mem.index(idx, ptr=True).load().cast(dtypes.uint32) | (mem.index(idx + _const(dtypes.index, 1), ptr=True).load().cast(dtypes.uint32) << _u32(8))).cast(dt)
      else:
        val = _u32(0)
        for i in range(4): val = val | (mem.index(idx + _const(dtypes.index, i), ptr=True).load().cast(dtypes.uint32) << _u32(i * 8))
    else:
      idx = (addr >> _const(addr.dtype, 2)).cast(dtypes.index)
      idx = idx.valid(active) if active is not None else idx
      val = mem.index(idx)
      if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
        idx2 = ((addr + _const(adt, 4)) >> _const(adt, 2)).cast(dtypes.index)
        idx2 = idx2.valid(active) if active is not None else idx2
        val = val.cast(dtypes.uint64) | (mem.index(idx2).cast(dtypes.uint64) << _u64(32))
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

def _tok_str(toks: list[Token]) -> str: return ' '.join(t.val for t in toks)

# Unified block parser for pcode
def _subst_loop_var(line: str, loop_var: str, val: int) -> str:
  """Substitute loop variable and evaluate bracket expressions.
  Converts var[loop_var] to var{val} for array element access (like the old regex parser)."""
  toks = tokenize(line)
  # First pass: convert var[loop_var] to var{loop_var} to mark for array element assignment
  result_toks, j = [], 0
  while j < len(toks):
    t = toks[j]
    # Check for pattern: IDENT[loop_var] where it's not preceded by a dot (not .type[...])
    if t.type == 'IDENT' and j+3 < len(toks) and toks[j+1].type == 'LBRACKET' and toks[j+2].type == 'IDENT' and toks[j+2].val == loop_var and toks[j+3].type == 'RBRACKET':
      # Check that it's not .type[loop_var]
      if not result_toks or result_toks[-1].type != 'DOT':
        result_toks.append(t)
        result_toks.append(Token('LBRACE', '{'))
        result_toks.append(Token('NUM', str(val)))
        result_toks.append(Token('RBRACE', '}'))
        j += 4
        continue
    result_toks.append(t)
    j += 1
  # Second pass: substitute loop variable in remaining positions
  subst_parts = [str(val) if t.type == 'IDENT' and t.val == loop_var else t.val for t in result_toks if t.type != 'EOF']
  return ' '.join(subst_parts)

def parse_block(lines: list[str], start: int, vars: dict[str, UOp], funcs: dict | None = None,
                assigns: list | None = None) -> tuple[int, dict[str, UOp], UOp | None]:
  """Parse a block of pcode. Returns (next_line, block_assigns, return_value).
  If assigns list is provided, side effects (MEM/VGPR writes) are appended to it."""
  if funcs is None: funcs = _FUNCS
  block_assigns: dict[str, UOp] = {}
  i = start
  def ctx(): return {**vars, **block_assigns}

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
      return i + 1, block_assigns, parse_expr(rest, ctx(), funcs)

    # for loop
    if first == 'for':
      # Parse: for VAR in [SIZE']START : [SIZE']END do
      p = Parser(toks, vars, funcs)
      p.eat('IDENT')  # for
      loop_var = p.eat('IDENT').val
      p.eat('IDENT')  # in
      if p.at('NUM') and p.peek(1).type == 'QUOTE': p.eat('NUM'); p.eat('QUOTE')
      if p.at('NUM'):
        start_val = int(p.eat('NUM').val.rstrip('UuLl'))
      else:
        start_expr = p.expr_top()
        start_val = int(start_expr.arg) if start_expr.op == Ops.CONST else 0
      p.eat('COLON')
      if p.at('NUM') and p.peek(1).type == 'QUOTE': p.eat('NUM'); p.eat('QUOTE')
      if p.at('NUM'):
        end_val = int(p.eat('NUM').val.rstrip('UuLl'))
      else:
        end_expr = p.expr_top()
        end_val = int(end_expr.arg) if end_expr.op == Ops.CONST else 0
      # Collect body
      i += 1; body_lines, depth = [], 1
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
          found = block_assigns.get(found_var, vars.get(found_var))
          not_found = found.eq(_const(dtypes.bool, False))
          for var, val in iter_assigns.items():
            if var != found_var:
              old = block_assigns.get(var, vars.get(var, _u32(0)))
              block_assigns[var] = vars[var] = not_found.where(val, old.cast(val.dtype) if val.dtype != old.dtype and val.dtype.itemsize == old.dtype.itemsize else old)
          for j, bl in enumerate(body_lines):
            bl_l = bl.strip().lower()
            if bl_l.startswith('if ') and bl_l.endswith(' then'):
              if any(body_lines[k].strip().lower() == 'break' for k in range(j+1, len(body_lines))):
                cond_str = _subst_loop_var(bl.strip()[3:-5].strip(), loop_var, loop_i)
                cond = _to_bool(parse_expr(cond_str, {**vars, **block_assigns}, funcs))
                block_assigns[found_var] = vars[found_var] = not_found.where(cond, found)
                break
        else:
          block_assigns.update(iter_assigns); vars.update(iter_assigns)
      continue

    # declare
    if first == 'declare':
      if '[' not in line and len(toks) >= 2 and toks[1].type == 'IDENT': vars[toks[1].val] = _u32(0)
      i += 1; continue

    # lambda definition
    if first != '{' and '=' in line and 'lambda' in line and any(t.type == 'IDENT' and t.val == 'lambda' for t in toks):
      name = toks[0].val
      body_start, depth = line[line.find('(', line.find('lambda')):], 0
      params_end = 0
      for j, ch in enumerate(body_start):
        if ch == '(': depth += 1
        elif ch == ')':
          depth -= 1
          if depth == 0: params_end = j + 1; break
      params = [p.strip() for p in body_start[1:params_end-1].split(',') if p.strip()]
      rest = body_start[params_end:].strip()
      if rest.startswith('('):
        depth, body_end = 1, 1
        for j, ch in enumerate(rest[1:], 1):
          if ch == '(': depth += 1
          elif ch == ')':
            depth -= 1
            if depth == 0: body_end = j; break
        body = rest[1:body_end].strip()
        if depth > 0:
          body_lines_lst = [rest[1:]]
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
        else: i += 1
        vars[name] = ('lambda', params, body)
        continue

    # MEM assignment: MEM[addr].type (+|-)?= value
    if first == 'mem' and toks[1].type == 'LBRACKET':
      j, addr_toks = _match_bracket(toks, 1)
      addr = parse_expr(_tok_str(addr_toks), ctx(), funcs)
      if j < len(toks) and toks[j].type == 'DOT': j += 1
      dt_name = toks[j].val if j < len(toks) and toks[j].type == 'IDENT' else 'u32'
      dt, j = DTYPES.get(dt_name, dtypes.uint32), j + 1
      compound_op = None
      if j < len(toks) and toks[j].type == 'ASSIGN_OP': compound_op = toks[j].val; j += 1
      elif j < len(toks) and toks[j].type == 'EQUALS': j += 1
      rhs = parse_expr(_tok_str(toks[j:]), ctx(), funcs)
      if compound_op:
        mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')
        if mem is not None:
          adt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
          idx = (addr >> _const(adt, 2)).cast(dtypes.index)
          old = mem.index(idx)
          if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
            old = old.cast(dtypes.uint64) | (mem.index(((addr + _const(adt, 4)) >> _const(adt, 2)).cast(dtypes.index)).cast(dtypes.uint64) << _u64(32))
          rhs = (old + rhs) if compound_op == '+=' else (old - rhs)
      if assigns is not None: assigns.append((f'MEM[{_tok_str(addr_toks)}].{dt_name}', (addr, rhs)))
      i += 1; continue

    # VGPR assignment: VGPR[lane][reg] = value
    if first == 'vgpr' and toks[1].type == 'LBRACKET':
      j, lane_toks = _match_bracket(toks, 1)
      if j < len(toks) and toks[j].type == 'LBRACKET':
        j, reg_toks = _match_bracket(toks, j)
        if j < len(toks) and toks[j].type == 'EQUALS': j += 1
        ln, rg, val = parse_expr(_tok_str(lane_toks), ctx(), funcs), parse_expr(_tok_str(reg_toks), ctx(), funcs), parse_expr(_tok_str(toks[j:]), ctx(), funcs)
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
          val_str = ' '.join(t.val for t in toks[j:] if t.type != 'EOF')
          val = parse_expr(val_str, ctx(), funcs)
          lo_dt, hi_dt = DTYPES.get(lo_type, dtypes.uint64), DTYPES.get(hi_type, dtypes.uint32)
          lo_bits = 64 if lo_dt in (dtypes.uint64, dtypes.int64) else 32
          lo_val = val.cast(lo_dt) if val.dtype.itemsize * 8 <= lo_bits else (val & _const(val.dtype, (1 << lo_bits) - 1)).cast(lo_dt)
          hi_val = (val >> _const(val.dtype, lo_bits)).cast(hi_dt)
          block_assigns[lo_var] = vars[lo_var] = lo_val
          block_assigns[hi_var] = vars[hi_var] = hi_val
          if assigns is not None: assigns.extend([(f'{lo_var}.{lo_type}', lo_val), (f'{hi_var}.{hi_type}', hi_val)])
          i += 1; continue

    # Bit slice: var[hi:lo] = value or var.type[hi:lo] = value
    if len(toks) >= 5 and toks[0].type == 'IDENT' and (toks[1].type == 'LBRACKET' or (toks[1].type == 'DOT' and toks[3].type == 'LBRACKET')):
      bracket_start = 2 if toks[1].type == 'LBRACKET' else 4
      j = bracket_start
      colon_pos = None
      while j < len(toks) and toks[j].type != 'RBRACKET':
        if toks[j].type == 'COLON': colon_pos = j
        j += 1
      if colon_pos is not None:
        hi_str = ' '.join(t.val for t in toks[bracket_start:colon_pos] if t.type != 'EOF')
        lo_str = ' '.join(t.val for t in toks[colon_pos+1:j] if t.type != 'EOF')
        try:
          hi, lo = max(int(eval(hi_str)), int(eval(lo_str))), min(int(eval(hi_str)), int(eval(lo_str)))
          var = toks[0].val
          j += 1
          if j < len(toks) and toks[j].type == 'DOT': j += 2
          if j < len(toks) and toks[j].type == 'EQUALS': j += 1
          val_str = ' '.join(t.val for t in toks[j:] if t.type != 'EOF')
          val = parse_expr(val_str, ctx(), funcs)
          dt_suffix = toks[2].val if toks[1].type == 'DOT' else None
          if assigns is not None: assigns.append((f'{var}[{hi}:{lo}]' + (f'.{dt_suffix}' if dt_suffix else ''), val))
          if var not in vars: vars[var] = _const(dtypes.uint64 if hi >= 32 else dtypes.uint32, 0)
          old = block_assigns.get(var, vars.get(var))
          mask = _u32(((1 << (hi - lo + 1)) - 1) << lo)
          block_assigns[var] = vars[var] = (old & (mask ^ _u32(0xFFFFFFFF))) | (_val_to_bits(val) << _u32(lo))
          i += 1; continue
        except: pass

    # Array element: var{idx} = value
    if len(toks) >= 5 and toks[0].type == 'IDENT' and toks[1].type == 'LBRACE' and toks[2].type == 'NUM':
      var, idx = toks[0].val, int(toks[2].val)
      j = 4
      while j < len(toks) and toks[j].type != 'EQUALS': j += 1
      if j < len(toks):
        val_str = ' '.join(t.val for t in toks[j+1:] if t.type != 'EOF')
        val = parse_expr(val_str, ctx(), funcs)
        existing = block_assigns.get(var, vars.get(var))
        if existing is not None and isinstance(existing, UOp):
          block_assigns[var] = vars[var] = _set_bit(existing, _u32(idx), val)
        else:
          block_assigns[f'{var}{idx}'] = vars[f'{var}{idx}'] = val
        i += 1; continue

    # Compound assignment: var += or var -=
    for j, t in enumerate(toks):
      if t.type == 'ASSIGN_OP':
        var = toks[0].val
        old = block_assigns.get(var, vars.get(var, _u32(0)))
        rhs_str = ' '.join(tk.val for tk in toks[j+1:] if tk.type != 'EOF')
        rhs = parse_expr(rhs_str, ctx(), funcs)
        if rhs.dtype != old.dtype: rhs = rhs.cast(old.dtype)
        block_assigns[var] = vars[var] = (old + rhs) if t.val == '+=' else (old - rhs)
        i += 1; break
    else:
      # Typed element: var.type[idx] = value
      if len(toks) >= 7 and toks[0].type == 'IDENT' and toks[1].type == 'DOT' and toks[2].type == 'IDENT' and toks[3].type == 'LBRACKET' and toks[4].type == 'NUM':
        var, dt_name, idx = toks[0].val, toks[2].val, int(toks[4].val)
        dt = DTYPES.get(dt_name, dtypes.uint32)
        j = 6
        while j < len(toks) and toks[j].type != 'EQUALS': j += 1
        if j < len(toks):
          val_str = ' '.join(t.val for t in toks[j+1:] if t.type != 'EOF')
          val, old = parse_expr(val_str, ctx(), funcs), block_assigns.get(var, vars.get(var, _u32(0)))
          bw, lo_bit = dt.itemsize * 8, idx * dt.itemsize * 8
          mask = _u32(((1 << bw) - 1) << lo_bit)
          block_assigns[var] = vars[var] = (old & (mask ^ _u32(0xFFFFFFFF))) | (((val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val) & _u32((1 << bw) - 1)) << _u32(lo_bit))
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
          bit_expr_str = ' '.join(t.val for t in toks[4:j-1] if t.type != 'EOF')
          bit_pos = _to_u32(parse_expr(bit_expr_str, ctx(), funcs))
          while j < len(toks) and toks[j].type != 'EQUALS': j += 1
          if j < len(toks):
            val_str = ' '.join(t.val for t in toks[j+1:] if t.type != 'EOF')
            val = parse_expr(val_str, ctx(), funcs)
            old, mask = block_assigns.get(var, vars.get(var, _u32(0))), _u32(1) << bit_pos
            block_assigns[var] = vars[var] = (old | mask) if val.op == Ops.CONST and val.arg == 1 else \
                                              (old & (mask ^ _u32(0xFFFFFFFF))) if val.op == Ops.CONST and val.arg == 0 else _set_bit(old, bit_pos, val)
            i += 1; continue

      # Bit index: var[expr] = value (bit assignment to existing scalar)
      if len(toks) >= 5 and toks[0].type == 'IDENT' and toks[1].type == 'LBRACKET':
        var = toks[0].val
        existing = block_assigns.get(var, vars.get(var))
        if existing is not None and isinstance(existing, UOp) and not any(f'{var}{k}' in vars or f'{var}{k}' in block_assigns for k in range(8)):
          j = 2
          while j < len(toks) and toks[j].type != 'RBRACKET': j += 1
          bit_str = ' '.join(t.val for t in toks[2:j] if t.type != 'EOF')
          j += 1
          while j < len(toks) and toks[j].type != 'EQUALS': j += 1
          if j < len(toks):
            val_str = ' '.join(t.val for t in toks[j+1:] if t.type != 'EOF')
            block_assigns[var] = vars[var] = _set_bit(existing, _to_u32(parse_expr(bit_str, ctx(), funcs)), parse_expr(val_str, ctx(), funcs))
            i += 1; continue

      # If/elsif/else - skip branches with statically false conditions (WAVE32/WAVE64)
      if first == 'if':
        def parse_cond(s, kw):
          ll = s.lower()
          return _to_bool(parse_expr(s[ll.find(kw) + len(kw):ll.rfind('then')].strip(), ctx(), funcs))
        def not_static_false(c): return c.op != Ops.CONST or c.arg is not False
        cond = parse_cond(line, 'if')
        conditions, else_branch, vars_snap = ([(cond, None)] if not_static_false(cond) else []), (None, {}), dict(vars)
        i += 1
        i, branch, ret = parse_block(lines, i, vars, funcs, assigns)
        if conditions: conditions[0] = (cond, ret if ret is not None else branch)
        vars.clear(); vars.update(vars_snap)
        while i < len(lines):
          ltoks = tokenize(lines[i])
          if ltoks[0].type != 'IDENT': break
          lf = ltoks[0].val.lower()
          if lf == 'elsif':
            c = parse_cond(lines[i], 'elsif')
            i += 1; i, branch, ret = parse_block(lines, i, vars, funcs, assigns)
            if not_static_false(c): conditions.append((c, ret if ret is not None else branch))
            vars.clear(); vars.update(vars_snap)
          elif lf == 'else':
            i += 1; i, branch, ret = parse_block(lines, i, vars, funcs, assigns)
            else_branch = (ret, branch)
            vars.clear(); vars.update(vars_snap)
          elif lf == 'endif': i += 1; break
          else: break
        # Check if any branch returned a value (lambda-style)
        if any(isinstance(br, UOp) for _, br in conditions):
          result = else_branch[0]
          for c, rv in reversed(conditions):
            if rv is not None:
              if rv.dtype != result.dtype and rv.dtype.itemsize == result.dtype.itemsize: result = result.cast(rv.dtype)
              result = c.where(rv, result)
          return i, block_assigns, result
        # Main style: merge variable assignments with WHERE
        else_assigns = else_branch[1]
        all_vars = set().union(*[ba.keys() for _, ba in conditions], else_assigns.keys())
        for var in all_vars:
          result = else_assigns.get(var, block_assigns.get(var, vars.get(var, _u32(0))))
          for cond, ba in reversed(conditions):
            if var in ba:
              tv = ba[var]
              result = cond.where(tv, result.cast(tv.dtype) if tv.dtype != result.dtype and tv.dtype.itemsize == result.dtype.itemsize else result)
          block_assigns[var] = vars[var] = result
        continue

      # Regular assignment: var = value
      for j, t in enumerate(toks):
        if t.type == 'EQUALS':
          if any(toks[k].type == 'OP' and toks[k].val in ('<', '>', '!', '=') for k in range(j)): break
          base_var = toks[0].val
          rhs_str = ' '.join(tk.val for tk in toks[j+1:] if tk.type != 'EOF')
          block_assigns[base_var] = vars[base_var] = parse_expr(rhs_str, ctx(), funcs)
          i += 1; break
      else: i += 1
      continue
    continue
  return i, block_assigns, None

def _parse_lambda_body(body: str, vars: dict[str, UOp], funcs: dict) -> UOp:
  lines = [l.strip() for l in body.replace(';', '\n').split('\n') if l.strip() and not l.strip().startswith('//')]
  _, _, result = parse_block(lines, 0, vars, funcs)
  return result if result is not None else _u32(0)

# Built-in function registry
_FUNCS: dict[str, callable] = {}

def _register_funcs():
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

  def _signext(a):
    val = a[0]
    for bits, mask, ext in [(8, 0xFF, 0xFFFFFF00), (16, 0xFFFF, 0xFFFF0000)]:
      if (val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == mask) or val.dtype.itemsize == bits // 8:
        v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
        sb = (v32 >> _u32(bits - 1)) & _u32(1)
        return sb.ne(_u32(0)).where(v32 | _u32(ext), v32).cast(dtypes.int)
    return val.cast(dtypes.int64) if val.dtype in (dtypes.int, dtypes.int32) else val

  def _abs(a):
    if a[0].dtype not in (dtypes.float32, dtypes.float64, dtypes.half): return a[0]
    _, _, _, _, shift = _float_info(a[0])
    sign_mask = {10: 0x7FFF, 23: 0x7FFFFFFF, 52: 0x7FFFFFFFFFFFFFFF}[shift]
    bt, ft = {10: (dtypes.uint16, dtypes.half), 23: (dtypes.uint32, dtypes.float32), 52: (dtypes.uint64, dtypes.float64)}[shift]
    return (a[0].bitcast(bt) & _const(bt, sign_mask)).bitcast(ft)

  def _f_to_u(f, dt): return UOp(Ops.TRUNC, f.dtype, ((f < _const(f.dtype, 0.0)).where(_const(f.dtype, 0.0), f),)).cast(dt)

  def _cvt_quiet(a):
    bits, _, _, qb, _ = _float_info(a[0])
    bt, ft = (dtypes.uint64, dtypes.float64) if a[0].dtype == dtypes.float64 else (dtypes.uint16, dtypes.half) if a[0].dtype == dtypes.half else (dtypes.uint32, dtypes.float32)
    return (a[0].bitcast(bt) | qb).bitcast(ft)

  def _is_denorm(a):
    bits, exp_m, mant_m, _, _ = _float_info(a[0])
    return (bits & exp_m).eq(_const(bits.dtype, 0)) & (bits & mant_m).ne(_const(bits.dtype, 0))

  _EXP_BITS = {10: 0x1F, 23: 0xFF, 52: 0x7FF}
  def _get_exp(bits, shift): return ((bits >> _const(bits.dtype, shift)) & _const(bits.dtype, _EXP_BITS[shift])).cast(dtypes.int)

  def _exponent(a):
    bits, _, _, _, shift = _float_info(a[0])
    return _get_exp(bits, shift)

  def _div_would_be_denorm(a):
    bits_n, _, _, _, shift = _float_info(a[0])
    bits_d, _, _, _, _ = _float_info(a[1])
    min_exp = {10: -14, 23: -126, 52: -1022}[shift]
    return (_get_exp(bits_n, shift) - _get_exp(bits_d, shift)) < _const(dtypes.int, min_exp)

  def _sign(a):
    bits, _, _, _, shift = _float_info(a[0])
    sign_shift = {10: 15, 23: 31, 52: 63}[shift]
    return ((bits >> _const(bits.dtype, sign_shift)) & _const(bits.dtype, 1)).cast(dtypes.uint32)

  def _signext_from_bit(a):
    val, w = a[0], a[1]
    is_64bit = val.dtype in (dtypes.uint64, dtypes.int64)
    dt = dtypes.uint64 if is_64bit else dtypes.uint32
    mask_all = _const(dt, 0xFFFFFFFFFFFFFFFF if is_64bit else 0xFFFFFFFF)
    one = _const(dt, 1)
    val_u = val.cast(dt) if val.dtype != dt else val
    w_val = w.cast(dt) if w.dtype != dt else w
    sign_bit = (val_u >> (w_val - one)) & one
    ext_mask = ((one << w_val) - one) ^ mask_all
    return sign_bit.ne(_const(dt, 0)).where(val_u | ext_mask, val_u)

  def _ldexp(a):
    val, exp = a[0], a[1]
    if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if exp.dtype in (dtypes.uint32, dtypes.uint64): exp = exp.cast(dtypes.int if exp.dtype == dtypes.uint32 else dtypes.int64)
    return val * UOp(Ops.EXP2, val.dtype, (exp.cast(val.dtype),))

  def _frexp_mant(a):
    val = a[0].bitcast(dtypes.float32) if a[0].dtype == dtypes.uint32 else a[0].bitcast(dtypes.float64) if a[0].dtype == dtypes.uint64 else a[0]
    if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) & _u32(0x807FFFFF)) | _u32(0x3f000000)).bitcast(dtypes.float32)
    return ((val.bitcast(dtypes.uint64) & _const(dtypes.uint64, 0x800FFFFFFFFFFFFF)) | _const(dtypes.uint64, 0x3fe0000000000000)).bitcast(dtypes.float64)

  def _frexp_exp(a):
    val = a[0].bitcast(dtypes.float32) if a[0].dtype == dtypes.uint32 else a[0].bitcast(dtypes.float64) if a[0].dtype == dtypes.uint64 else a[0]
    if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) >> _u32(23)) & _u32(0xFF)).cast(dtypes.int) - _const(dtypes.int, 126)
    return ((val.bitcast(dtypes.uint64) >> _const(dtypes.uint64, 52)) & _const(dtypes.uint64, 0x7FF)).cast(dtypes.int) - _const(dtypes.int, 1022)

  TWO_OVER_PI = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6
  # TWO_OVER_PI as 19 u64 words for trig_preop_result (word[0] = bits 0-63, word[18] = bits 1152-1200)
  _PREOP_WORDS = tuple((TWO_OVER_PI >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(19))
  def _trig_preop(a):
    # Extract 53 bits from position (1148 - shift) in the 1201-bit 2/PI constant
    # Using word-based selection: 19 conditions instead of 1149
    shift = a[0].cast(dtypes.uint32)
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

  def _ff1(a, bits):
    dt = dtypes.uint64 if bits == 64 else dtypes.uint32
    val = a[0].cast(dt) if a[0].dtype != dt else a[0]
    result = _const(dtypes.int, -1)
    for i in range(bits):
      cond = ((val >> _const(dt, i)) & _const(dt, 1)).ne(_const(dt, 0)) & result.eq(_const(dtypes.int, -1))
      result = cond.where(_const(dtypes.int, i), result)
    return result

  _FUNCS.update({
    'sqrt': lambda a: UOp(Ops.SQRT, a[0].dtype, (a[0],)), 'trunc': lambda a: UOp(Ops.TRUNC, a[0].dtype, (a[0],)),
    'log2': lambda a: UOp(Ops.LOG2, a[0].dtype, (a[0],)), 'sin': lambda a: _trig_reduce(a[0]),
    'cos': lambda a: _trig_reduce(a[0], 0.25), 'floor': lambda a: _floor(a[0]), 'fract': lambda a: a[0] - _floor(a[0]),
    'signext': lambda a: _signext(a), 'abs': lambda a: _abs(a),
    'isEven': lambda a: (UOp(Ops.TRUNC, a[0].dtype, (a[0],)).cast(dtypes.int) & _const(dtypes.int, 1)).eq(_const(dtypes.int, 0)),
    'max': lambda a: UOp(Ops.MAX, a[0].dtype, (a[0], a[1])),
    'min': lambda a: UOp(Ops.MAX, a[0].dtype, (a[0].neg(), a[1].neg())).neg(),
    'pow': lambda a: UOp(Ops.EXP2, dtypes.float32, (a[1].bitcast(dtypes.float32),)),
    'fma': lambda a: a[0] * a[1] + a[2],
    'i32_to_f32': lambda a: a[0].cast(dtypes.int).cast(dtypes.float32),
    'u32_to_f32': lambda a: a[0].cast(dtypes.uint32).cast(dtypes.float32),
    'f32_to_i32': lambda a: UOp(Ops.TRUNC, dtypes.float32, (a[0].bitcast(dtypes.float32),)).cast(dtypes.int),
    'f32_to_u32': lambda a: _f_to_u(a[0].bitcast(dtypes.float32), dtypes.uint32),
    'f64_to_i32': lambda a: UOp(Ops.TRUNC, dtypes.float64, (a[0].bitcast(dtypes.float64),)).cast(dtypes.int),
    'f64_to_u32': lambda a: _f_to_u(a[0].bitcast(dtypes.float64), dtypes.uint32),
    'f16_to_f32': lambda a: _f16_extract(a[0]).cast(dtypes.float32),
    'f32_to_f16': lambda a: a[0].cast(dtypes.half),
    'f32_to_f64': lambda a: a[0].bitcast(dtypes.float32).cast(dtypes.float64),
    'f64_to_f32': lambda a: a[0].bitcast(dtypes.float64).cast(dtypes.float32),
    'i32_to_f64': lambda a: a[0].cast(dtypes.int).cast(dtypes.float64),
    'u32_to_f64': lambda a: a[0].cast(dtypes.uint32).cast(dtypes.float64),
    'f16_to_i16': lambda a: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a[0]),)).cast(dtypes.int16),
    'f16_to_u16': lambda a: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a[0]),)).cast(dtypes.uint16),
    'i16_to_f16': lambda a: a[0].cast(dtypes.int16).cast(dtypes.half),
    'u16_to_f16': lambda a: a[0].cast(dtypes.uint16).cast(dtypes.half),
    'bf16_to_f32': lambda a: (((a[0].cast(dtypes.uint32) if a[0].dtype != dtypes.uint32 else a[0]) & _u32(0xFFFF)) << _u32(16)).bitcast(dtypes.float32),
    'isNAN': lambda a: _isnan(a[0]), 'isSignalNAN': lambda a: _check_nan(a[0], False),
    'isQuietNAN': lambda a: _check_nan(a[0], True), 'cvtToQuietNAN': lambda a: _cvt_quiet(a),
    'isDENORM': lambda a: _is_denorm(a), 'exponent': lambda a: _exponent(a),
    'divWouldBeDenorm': lambda a: _div_would_be_denorm(a), 'sign': lambda a: _sign(a),
    'signext_from_bit': lambda a: _signext_from_bit(a), 'ldexp': lambda a: _ldexp(a),
    'frexp_mant': lambda a: _frexp_mant(a), 'mantissa': lambda a: _frexp_mant(a),
    'frexp_exp': lambda a: _frexp_exp(a), 'trig_preop_result': lambda a: _trig_preop(a),
    's_ff1_i32_b32': lambda a: _ff1(a, 32), 's_ff1_i32_b64': lambda a: _ff1(a, 64),
  })
  for is_max, name in [(False, 'min'), (True, 'max')]:
    for dt, sfx in [(dtypes.float32, 'f32'), (dtypes.int, 'i32'), (dtypes.uint32, 'u32'), (dtypes.int16, 'i16'), (dtypes.uint16, 'u16')]:
      _FUNCS[f'v_{name}_{sfx}'] = lambda a, im=is_max, d=dt: _minmax_reduce(im, d, a)
      _FUNCS[f'v_{name}3_{sfx}'] = lambda a, im=is_max, d=dt: _minmax_reduce(im, d, a)

_register_funcs()

def parse_expr(expr: str, vars: dict, funcs: dict | None = None) -> UOp:
  return Parser(tokenize(expr.strip().rstrip(';')), vars, funcs).parse()

