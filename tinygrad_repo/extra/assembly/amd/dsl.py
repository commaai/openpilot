# dsl.py - clean DSL for AMD assembly
from typing import Any

# ══════════════════════════════════════════════════════════════
# Registers - unified src encoding space (0-511)
# ══════════════════════════════════════════════════════════════

class Reg:
  # Register names vary by arch: RDNA has NULL@124/M0@125, CDNA has M0@124/reserved@125
  # RDNA4 has DPP8@233, CDNA has SDWA@249/DPP@250/VCCZ@251/EXECZ@252
  _NAMES = {102: "FLAT_SCRATCH_LO", 103: "FLAT_SCRATCH_HI", 104: "XNACK_MASK_LO", 105: "XNACK_MASK_HI",
            106: "VCC_LO", 107: "VCC_HI", 124: "NULL", 125: "M0", 126: "EXEC_LO", 127: "EXEC_HI",
            233: "DPP8", 234: "DPP8FI", 235: "SHARED_BASE", 236: "SHARED_LIMIT", 237: "PRIVATE_BASE", 238: "PRIVATE_LIMIT",
            240: "0.5", 241: "-0.5", 242: "1.0", 243: "-1.0", 244: "2.0", 245: "-2.0", 246: "4.0", 247: "-4.0",
            248: "INV_2PI", 249: "SDWA", 250: "DPP", 251: "VCCZ", 252: "EXECZ", 253: "SCC", 254: "SRC_LDS_DIRECT", 255: "LIT"}
  _PAIRS = {106: "VCC", 126: "EXEC"}

  def __init__(self, offset: int = 0, sz: int = 512, *, neg: bool = False, abs_: bool = False, hi: bool = False):
    self.offset, self.sz = offset, sz
    self.neg, self.abs_, self.hi = neg, abs_, hi

  def __hash__(self): return hash((self.offset, self.sz, self.neg, self.abs_, self.hi))
  def __getitem__(self, key):
    if isinstance(key, slice):
      start, stop = key.start or 0, key.stop or (self.sz - 1)
      if start < 0 or stop >= self.sz: raise RuntimeError(f"slice [{start}:{stop}] out of bounds for size {self.sz}")
      return Reg(self.offset + start, stop - start + 1)
    if key < 0 or key >= self.sz: raise RuntimeError(f"index {key} out of bounds for size {self.sz}")
    return Reg(self.offset + key, 1)
  def __eq__(self, other):
    if isinstance(other, Reg):
      return (self.offset == other.offset and self.sz == other.sz and
              self.neg == other.neg and self.abs_ == other.abs_ and self.hi == other.hi)
    return NotImplemented
  def __add__(self, other):
    if isinstance(other, int): return Reg(self.offset + other, self.sz)
    return NotImplemented
  def __neg__(self) -> 'Reg': return Reg(self.offset, self.sz, neg=not self.neg, abs_=self.abs_, hi=self.hi)
  def __abs__(self) -> 'Reg': return Reg(self.offset, self.sz, neg=self.neg, abs_=True, hi=self.hi)
  @property
  def h(self) -> 'Reg': return Reg(self.offset, self.sz, neg=self.neg, abs_=self.abs_, hi=True)
  @property
  def l(self) -> 'Reg': return Reg(self.offset, self.sz, neg=self.neg, abs_=self.abs_, hi=False)
  def fmt(self, sz=None, parens=False, upper=False) -> str:
    o, sz = self.offset, sz or self.sz
    l, r = ("[", "]") if parens or sz > 1 else ("", "")  # brackets for multi-reg or when parens=True
    if 256 <= o < 512: idx = o - 256; base = f"v{l}{idx}{r}" if sz == 1 else f"v[{idx}:{idx + sz - 1}]"
    elif o < 106: base = f"s{l}{o}{r}" if sz == 1 else f"s[{o}:{o + sz - 1}]"
    elif sz == 2 and o in self._PAIRS: base = self._PAIRS[o] if upper else self._PAIRS[o].lower()
    elif o in self._NAMES: base = self._NAMES[o] if upper else self._NAMES[o].lower()  # special regs (any sz)
    elif 108 <= o < 124: idx = o - 108; base = f"ttmp{l}{idx}{r}" if sz == 1 else f"ttmp[{idx}:{idx + sz - 1}]"
    elif 128 <= o <= 192: base = str(o - 128)  # inline int constants (0-64)
    elif 193 <= o <= 208: base = str(-(o - 192))  # inline negative int constants (-1 to -16)
    else: raise RuntimeError(f"unknown register: offset={o}, sz={sz}")
    if self.hi: base += ".h"
    if self.abs_: base = f"abs({base})" if upper else f"|{base}|"
    if self.neg: base = f"-{base}"
    return base
  def __repr__(self): return self.fmt(parens=True, upper=True)

# Full src encoding space
src = Reg(0, 512)

# Slices for each region (inclusive end)
s = src[0:105]           # SGPR0-105
VCC_LO = src[106]
VCC_HI = src[107]
VCC = src[106:107]
ttmp = src[108:123]      # TTMP0-15
NULL = OFF = src[124]
M0 = src[125]
EXEC_LO = src[126]
EXEC_HI = src[127]
EXEC = src[126:127]
# 128: 0, 129-192: integers 1-64, 193-208: integers -1 to -16
# 240-248: float constants (0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2*PI))
INV_2PI = src[248]
SDWA = src[249]
DPP = DPP16 = src[250]
VCCZ = src[251]
EXECZ = src[252]
SCC = src[253]
SRC_LDS_DIRECT = src[254]
LIT = src[255]           # literal constant marker
v = src[256:511]         # VGPR0-255

# ══════════════════════════════════════════════════════════════
# BitField
# ══════════════════════════════════════════════════════════════

class _Bits:
  """Helper for defining bit fields with slice syntax: bits[hi:lo] or bits[n]."""
  def __getitem__(self, key) -> 'BitField': return BitField(key.start, key.stop) if isinstance(key, slice) else BitField(key, key)
bits = _Bits()

class BitField:
  name: str | None
  def __init__(self, hi: int, lo: int, default: int = 0):
    self.hi, self.lo, self.default, self.name, self.mask = hi, lo, default, None, (1 << (hi - lo + 1)) - 1
  def __set_name__(self, owner, name: str): self.name = name
  def __eq__(self, other) -> 'FixedBitField':  # type: ignore[override]
    if isinstance(other, int): return FixedBitField(self.hi, self.lo, other)
    raise TypeError(f"BitField.__eq__ expects int, got {type(other).__name__}")
  def enum(self, enum_cls) -> 'EnumBitField': return EnumBitField(self.hi, self.lo, enum_cls)
  def encode(self, val) -> int:
    assert isinstance(val, int), f"BitField.encode expects int, got {type(val).__name__}"
    return val
  def decode(self, val): return val
  def set(self, raw: int, val) -> int:
    if val is None: val = self.default
    encoded = self.encode(val)
    # Handle signed values: convert negative to 2's complement
    if encoded < 0: encoded = encoded & self.mask
    if encoded < 0 or encoded > self.mask: raise RuntimeError(f"field '{self.name}': value {encoded} doesn't fit in {self.hi - self.lo + 1} bits")
    return (raw & ~(self.mask << self.lo)) | (encoded << self.lo)
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    return self.decode((obj._raw >> self.lo) & self.mask)
  def __set__(self, obj, val): obj._raw = self.set(obj._raw, val)

class FixedBitField(BitField):
  def set(self, raw: int, val=None) -> int:
    assert val is None, f"FixedBitField does not accept values, got {val}"
    return super().set(raw, self.default)

class EnumBitField(BitField):
  def __init__(self, hi: int, lo: int, enum_cls, allowed: set | None = None):
    super().__init__(hi, lo)
    self._enum = enum_cls
    self.allowed = allowed  # if set, only these enum values are valid for this encoding
  def encode(self, val) -> int:
    if not isinstance(val, self._enum): raise RuntimeError(f"expected {self._enum.__name__}, got {type(val).__name__}")
    if self.allowed is not None and val not in self.allowed:
      raise RuntimeError(f"opcode {val.name} not allowed in this encoding")
    return val.value
  def decode(self, raw): return self._enum(raw)

# ══════════════════════════════════════════════════════════════
# Typed fields
# ══════════════════════════════════════════════════════════════

import struct
def _f32(f: float) -> int: return struct.unpack('I', struct.pack('f', f))[0]

class SrcField(BitField):
  _valid_range = (0, 511)  # inclusive
  _FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}

  def __init__(self, hi: int, lo: int, default=s[0]):
    super().__init__(hi, lo, default)
    expected_size = self._valid_range[1] - self._valid_range[0] + 1
    actual_size = 1 << (hi - lo + 1)
    if actual_size != expected_size:
      raise RuntimeError(f"{self.__class__.__name__}: field size {hi - lo + 1} bits ({actual_size}) doesn't match range {self._valid_range} ({expected_size})")

  def encode(self, val) -> int:
    """Encode value. Returns 255 (literal marker) for out-of-range values."""
    if isinstance(val, Reg): offset = val.offset
    elif isinstance(val, float): offset = self._FLOAT_ENC.get(val, 255)
    elif isinstance(val, int) and 0 <= val <= 64: offset = 128 + val
    elif isinstance(val, int) and -16 <= val < 0: offset = 192 - val
    elif isinstance(val, int): offset = 255  # literal
    else: raise TypeError(f"invalid src value {val}")
    if not (self._valid_range[0] <= offset <= self._valid_range[1]):
      raise TypeError(f"{self.__class__.__name__}: {val} (offset {offset}) out of range {self._valid_range}")
    return offset - self._valid_range[0]

  def decode(self, raw): return src[raw + self._valid_range[0]]

  def __get__(self, obj, objtype=None):
    if obj is None: return self
    reg = self.decode((obj._raw >> self.lo) & self.mask)
    # Resize register based on operand info (skip non-resizable special registers)
    # VCC/EXEC pairs (106, 126), NULL (124), M0 (125), float constants (240-255)
    if reg.offset not in (124, 125) and not 240 <= reg.offset <= 255:
      # Map variant field names (vsrc0->src0, vsrc1->src1, etc.) for DPP/SDWA classes
      assert self.name is not None
      name = self.name[1:] if self.name.startswith('v') and self.name[1:] in obj.op_regs else self.name
      if sz := obj.op_regs.get(name, 1): reg = Reg(reg.offset, sz, neg=reg.neg, abs_=reg.abs_, hi=reg.hi)
    return reg

class VGPRField(SrcField):
  _valid_range = (256, 511)
  def __init__(self, hi: int, lo: int, default=v[0]): super().__init__(hi, lo, default)
  def encode(self, val) -> int:
    if not isinstance(val, Reg): raise TypeError(f"VGPRField requires Reg, got {type(val).__name__}")
    # For 8-bit vdst fields in VOP1/VOP2 16-bit ops, bit 7 is opsel for dest half
    encoded = super().encode(val)
    if val.hi and (self.hi - self.lo + 1) == 8:
      if encoded >= 128:
        raise ValueError(f"VGPRField: v[{encoded}].h not encodable in 8-bit field (v[0:127] only for .h)")
      encoded |= 0x80
    return encoded
class SGPRField(SrcField): _valid_range = (0, 127)
class SSrcField(SrcField): _valid_range = (0, 255)

class AlignedSGPRField(BitField):
  """SGPR field with alignment requirement. Encoded as sgpr_index // alignment."""
  _align: int = 2
  def encode(self, val):
    if isinstance(val, int) and val == 0: return 0  # default: encode as s[0]
    if not isinstance(val, Reg): raise TypeError(f"{self.__class__.__name__} requires Reg, got {type(val).__name__}")
    if not (0 <= val.offset < 128): raise ValueError(f"{self.__class__.__name__} requires SGPR, got offset {val.offset}")
    if val.offset & (self._align - 1): raise ValueError(f"{self.__class__.__name__} requires {self._align}-aligned SGPR, got s[{val.offset}]")
    return val.offset >> (self._align.bit_length() - 1)
  def decode(self, raw): return src[raw << (self._align.bit_length() - 1)]
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    reg = self.decode((obj._raw >> self.lo) & self.mask)
    if sz := obj.op_regs.get(self.name, 1): reg = Reg(reg.offset, sz, neg=reg.neg, abs_=reg.abs_, hi=reg.hi)
    return reg

class SBaseField(AlignedSGPRField): _align = 2
class SRsrcField(AlignedSGPRField): _align = 4

class VDSTYField(BitField):
  """VOPD vdsty: encoded = vgpr_idx >> 1. Actual vgpr = (encoded << 1) | ((vdstx & 1) ^ 1)."""
  def encode(self, val):
    if not isinstance(val, Reg): raise TypeError(f"VDSTYField requires Reg, got {type(val).__name__}")
    if not (256 <= val.offset < 512): raise ValueError(f"VDSTYField requires VGPR, got offset {val.offset}")
    return (val.offset - 256) >> 1
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    raw = (obj._raw >> self.lo) & self.mask
    vdstx_bit0 = (obj.vdstx.offset - 256) & 1
    vgpr_idx = (raw << 1) | (vdstx_bit0 ^ 1)
    return Reg(256 + vgpr_idx, 1)

# ══════════════════════════════════════════════════════════════
# Operand info from XML
# ══════════════════════════════════════════════════════════════

import functools
from extra.assembly.amd.autogen.rdna3.operands import OPERANDS as OPERANDS_RDNA3
from extra.assembly.amd.autogen.rdna4.operands import OPERANDS as OPERANDS_RDNA4
from extra.assembly.amd.autogen.cdna.operands import OPERANDS as OPERANDS_CDNA
OPERANDS = {**OPERANDS_CDNA, **OPERANDS_RDNA3, **OPERANDS_RDNA4}

# ══════════════════════════════════════════════════════════════
# Inst base class
# ══════════════════════════════════════════════════════════════

def _needs_literal(val) -> bool:
  """Check if a value needs a literal constant (can't be encoded inline)."""
  if val is None or isinstance(val, Reg): return False
  if isinstance(val, float): return val not in SrcField._FLOAT_ENC
  if isinstance(val, int): return not (0 <= val <= 64 or -16 <= val < 0)
  return False

def _get_variant(cls, suffix: str):
  """Get a variant class by suffix (e.g., '_LIT') via module lookup."""
  import sys
  module = sys.modules.get(cls.__module__)
  return getattr(module, f"{cls.__name__}{suffix}", None) if module else None

def _canonical_name(name: str) -> str | None:
  """Map operand name to canonical name."""
  if name in ('src0', 'vsrc0', 'ssrc0'): return 's0'
  if name in ('src1', 'vsrc1', 'ssrc1'): return 's1'
  if name == 'src2': return 's2'
  if name in ('vdst', 'sdst', 'sdata'): return 'd'
  if name in ('data', 'vdata', 'data0', 'vsrc'): return 'data'
  return None

class Inst:
  _fields: list[tuple[str, BitField]]
  _base_size: int

  def __init_subclass__(cls):
    # Collect fields from all parent classes, then override with this class's fields
    inherited = {}
    for base in reversed(cls.__mro__[1:]):
      if hasattr(base, '_fields'):
        inherited.update({name: field for name, field in base._fields})
    inherited.update({name: val for name, val in cls.__dict__.items() if isinstance(val, BitField)})
    cls._fields = list(inherited.items())
    cls._base_size = (max(f.hi for _, f in cls._fields) + 8) // 8

  def __new__(cls, *args, **kwargs):
    # Auto-upgrade to variant if needed (only for base classes, not variants)
    if not any(cls.__name__.endswith(sfx) for sfx in ('_LIT', '_DPP16', '_DPP8', '_SDWA', '_SDWA_SDST', '_MFMA')):
      args_iter = iter(args)
      for name, field in cls._fields:
        if isinstance(field, FixedBitField): continue
        val = kwargs.get(name) if name in kwargs else next(args_iter, None)
        if not isinstance(field, SrcField): continue
        if isinstance(val, Reg) and val.offset == 255 and (lit_cls := _get_variant(cls, '_LIT')): return lit_cls(*args, **kwargs)
        if isinstance(val, Reg) and val.offset == 249:
          if (sdwa_cls := _get_variant(cls, '_SDWA') or _get_variant(cls, '_SDWA_SDST')): return sdwa_cls(*args, **kwargs)
        if isinstance(val, Reg) and val.offset == 250 and (dpp_cls := _get_variant(cls, '_DPP16')): return dpp_cls(*args, **kwargs)
        if _needs_literal(val) and (lit_cls := _get_variant(cls, '_LIT')): return lit_cls(*args, **kwargs)
    return object.__new__(cls)

  def __init__(self, *args, **kwargs):
    self._raw = 0
    # Map positional args to field names (skip FixedBitFields)
    args_iter = iter(args)
    vals: dict[str, Any] = {}
    for name, field in self._fields:
      if isinstance(field, FixedBitField): vals[name] = None
      elif name in kwargs: vals[name] = kwargs[name]
      else: vals[name] = next(args_iter, None)
    assert not (remaining := list(args_iter)), f"too many positional args: {remaining}"
    # Extract modifiers from Reg objects and merge into neg/abs/opsel
    neg_bits, abs_bits, opsel_bits = 0, 0, 0
    for name, bit in [('src0', 0), ('src1', 1), ('src2', 2)]:
      if name in vals and isinstance(vals[name], Reg):
        reg = vals[name]
        if reg.neg: neg_bits |= (1 << bit)
        if reg.abs_: abs_bits |= (1 << bit)
        if reg.hi: opsel_bits |= (1 << bit)
    if 'vdst' in vals and isinstance(vals['vdst'], Reg) and vals['vdst'].hi:
      opsel_bits |= (1 << 3)
    if neg_bits: vals['neg'] = (vals.get('neg') or 0) | neg_bits
    if abs_bits: vals['abs'] = (vals.get('abs') or 0) | abs_bits
    if opsel_bits: vals['opsel'] = (vals.get('opsel') or 0) | opsel_bits
    # For _LIT classes, capture literal value from SrcFields that encode to 255
    literal_val = None
    for name, field in self._fields:
      val = vals[name]
      if isinstance(field, SrcField) and val is not None and _needs_literal(val):
        literal_val = _f32(val) if isinstance(val, float) else val & 0xFFFFFFFF
    if literal_val is not None and 'literal' in vals:
      vals['literal'] = literal_val
    # Set all field values
    for name, field in self._fields:
      self._raw = field.set(self._raw, vals[name])
    # Validate register sizes against operand info (skip special registers like NULL, VCC, EXEC, SDWA/DPP markers)
    for name, expected in self.op_regs.items():
      if (val := vals.get(name)) is None: continue
      if isinstance(val, Reg) and val.sz != expected and not (106 <= val.offset <= 127 or 249 <= val.offset <= 255):
        raise TypeError(f"{name} expects {expected} register(s), got {val.sz}")

  @property
  def op_name(self) -> str: return getattr(self, 'op').name
  @property
  def operands(self) -> dict: return OPERANDS.get(getattr(self, 'op'), {}) if hasattr(self, 'op') else {}
  def _is_cdna(self) -> bool: return 'cdna' in type(self).__module__

  @functools.cached_property
  def op_bits(self) -> dict[str, int]:
    """Get bit widths for each operand field, with WAVE32 and addr/saddr adjustments."""
    if not hasattr(self, 'op'): return {k: v[1] for k, v in self.operands.items()}
    bits = {k: v[1] for k, v in self.operands.items()}
    # RDNA (WAVE32): condition masks, carry flags, and compare results are 32-bit
    if not self._is_cdna():
      name = self.op_name.lower()
      if 'cndmask' in name and 'src2' in bits: bits['src2'] = 32
      if '_co_ci_' in name and 'src2' in bits: bits['src2'] = 32  # carry-in source
      # VOP3SD: sdst is always wavefront-size dependent (carry-out or condition mask)
      if 'VOP3SD' in type(self).__name__ and 'sdst' in bits: bits['sdst'] = 32
      if 'cmp' in name and 'vdst' in bits: bits['vdst'] = 32
    # GLOBAL/FLAT: addr is 32-bit if saddr is valid SGPR, 64-bit if saddr is NULL
    # SCRATCH: addr is always 32-bit (offset from scratch base, not absolute address)
    if 'addr' in bits and (saddr_field := getattr(type(self), 'saddr', None)) and type(self).__name__ not in ('SCRATCH', 'VSCRATCH'):
      saddr_val = (self._raw >> saddr_field.lo) & saddr_field.mask  # access _raw directly to avoid recursion
      bits['addr'] = 64 if saddr_val in (124, 125) else 32  # 124=NULL, 125=M0
    # MUBUF/MTBUF: vaddr size depends on offen/idxen (1 or 2 regs)
    if 'vaddr' in bits and hasattr(self, 'offen') and hasattr(self, 'idxen'):
      bits['vaddr'] = max(1, self.offen + self.idxen) * 32
    # F8F6F4 MFMA: CBSZ selects matrix A format, BLGP selects matrix B format
    # VGPRs: FP8/BF8(0,1)=8, FP6/BF6(2,3)=6, FP4(4)=4
    if 'f8f6f4' in getattr(self, 'op_name', '').lower():
      # Use explicit fields if available (VOP3PX2), else extract from VOP3P-MAI bit positions
      cbsz = getattr(self, 'cbsz') if hasattr(type(self), 'cbsz') else (self._raw >> 8) & 0x7
      blgp = getattr(self, 'blgp') if hasattr(type(self), 'blgp') else (self._raw >> 61) & 0x7
      vgprs = {0: 8, 1: 8, 2: 6, 3: 6, 4: 4}
      bits['src0'], bits['src1'] = vgprs.get(cbsz, 8) * 32, vgprs.get(blgp, 8) * 32
    return bits
  @property
  def op_regs(self) -> dict[str, int]:
    """Get register counts for each operand field."""
    return {k: max(1, v // 32) for k, v in self.op_bits.items()}

  @functools.cached_property
  def canonical_op_bits(self) -> dict[str, int]:
    """Get bit widths with canonical names: {'s0', 's1', 's2', 'd', 'data'}."""
    bits = {'d': 32, 's0': 32, 's1': 32, 's2': 32, 'data': 32}
    for name, val in self.op_bits.items():
      if (cn := _canonical_name(name)): bits[cn] = val
    return bits

  @functools.cached_property
  def canonical_operands(self) -> dict:
    """Get operands with canonical names: {'s0', 's1', 's2', 'd', 'data'}."""
    result = {}
    for name, val in self.operands.items():
      if (cn := _canonical_name(name)): result[cn] = val
    return result

  @property
  def canonical_op_regs(self) -> dict[str, int]:
    """Get register counts with canonical names: {'s0', 's1', 's2', 'd', 'data'}."""
    return {k: max(1, v // 32) for k, v in self.canonical_op_bits.items()}

  def num_srcs(self) -> int:
    """Get number of source operands from operand info."""
    ops = self.operands
    if 'src2' in ops: return 3
    if 'src1' in ops or 'vsrc1' in ops or 'ssrc1' in ops: return 2
    if 'src0' in ops or 'vsrc0' in ops or 'ssrc0' in ops: return 1
    return 0
  @classmethod
  def _size(cls) -> int: return cls._base_size
  def size(self) -> int: return self._base_size
  def disasm(self) -> str:
    from extra.assembly.amd.disasm import disasm
    return disasm(self)

  def to_bytes(self) -> bytes: return self._raw.to_bytes(self._base_size, 'little')

  @property
  def _literal(self) -> int | None:
    """Get the literal value if this instruction has one."""
    return getattr(self, 'literal', None)

  def _variant_suffix(self) -> str | None:
    """Check if instruction needs a variant class (_LIT, _DPP8, _DPP16, _SDWA). Returns suffix or None."""
    cls_name = type(self).__name__
    # Don't check for variants if we're already a variant class
    if any(s in cls_name for s in ('_LIT', '_DPP8', '_DPP16', '_SDWA')): return None
    # VOPD: FMAMK/FMAAK opcodes always require literal (check by name since enum may differ across archs)
    for name in ('opx', 'opy'):
      if hasattr(self, name) and any(x in getattr(self, name).name for x in ('FMAMK', 'FMAAK')): return '_LIT'
    for name, field in self._fields:
      if isinstance(field, SrcField):
        off = getattr(self, name).offset
        if off == 255: return '_LIT'
        if off == 249: return '_SDWA' if self._is_cdna() else '_DPP8'
        if off == 250: return '_DPP16'
    return None

  @classmethod
  def from_bytes(cls, data: bytes):
    inst = object.__new__(cls)
    inst._raw = int.from_bytes(data[:cls._base_size], 'little')
    # Upgrade to variant class if needed (_LIT, _DPP8, _DPP16, _SDWA)
    if (suffix := inst._variant_suffix()) and (var_cls := _get_variant(cls, suffix)) is not None:
      return var_cls.from_bytes(data)
    return inst

  def __eq__(self, other): return type(self) is type(other) and self._raw == other._raw
  def __hash__(self): return hash((type(self), self._raw))

  def __repr__(self):
    # collect (repr, is_default) pairs, strip trailing defaults so repr roundtrips with eval
    name = self.op.name.lower() if hasattr(self, 'op') else type(self).__name__
    parts = [(repr(v := getattr(self, n)), v == f.default) for n, f in self._fields if n != 'op' and not isinstance(f, FixedBitField)]
    while parts and parts[-1][1]: parts.pop()
    return f"{name}({', '.join(p[0] for p in parts)})"
