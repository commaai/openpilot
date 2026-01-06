# library for RDNA3 assembly DSL
from __future__ import annotations
from enum import IntEnum
from typing import overload, Annotated, TypeVar, Generic

# Bit field DSL
class BitField:
  def __init__(self, hi: int, lo: int, name: str | None = None): self.hi, self.lo, self.name = hi, lo, name
  def __set_name__(self, owner, name): self.name, self._owner = name, owner
  def __eq__(self, val: int) -> tuple[BitField, int]: return (self, val)  # type: ignore
  def mask(self) -> int: return (1 << (self.hi - self.lo + 1)) - 1
  @property
  def marker(self) -> type | None:
    # Get marker from Annotated type hint if present
    import typing
    if hasattr(self, '_owner') and self.name:
      hints = typing.get_type_hints(self._owner, include_extras=True)
      if self.name in hints:
        hint = hints[self.name]
        if typing.get_origin(hint) is Annotated:
          args = typing.get_args(hint)
          return args[1] if len(args) > 1 else None
    return None
  @overload
  def __get__(self, obj: None, objtype: type) -> BitField: ...
  @overload
  def __get__(self, obj: object, objtype: type | None = None) -> int: ...
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    val = unwrap(obj._values.get(self.name, 0))
    # Convert to IntEnum if marker is an IntEnum subclass
    if self.marker and isinstance(self.marker, type) and issubclass(self.marker, IntEnum):
      try: return self.marker(val)
      except ValueError: pass
    return val

class _Bits:
  def __getitem__(self, key) -> BitField: return BitField(key.start, key.stop) if isinstance(key, slice) else BitField(key, key)
bits = _Bits()

# Register types
class Reg:
  def __init__(self, idx: int, count: int = 1, hi: bool = False): self.idx, self.count, self.hi = idx, count, hi
  def __repr__(self): return f"{self.__class__.__name__.lower()[0]}[{self.idx}]" if self.count == 1 else f"{self.__class__.__name__.lower()[0]}[{self.idx}:{self.idx + self.count}]"

T = TypeVar('T', bound=Reg)
class _RegFactory(Generic[T]):
  def __init__(self, cls: type[T], name: str): self._cls, self._name = cls, name
  @overload
  def __getitem__(self, key: int) -> Reg: ...
  @overload
  def __getitem__(self, key: slice) -> Reg: ...
  def __getitem__(self, key: int | slice) -> Reg:
    return self._cls(key.start, key.stop - key.start) if isinstance(key, slice) else self._cls(key)
  def __repr__(self): return f"<{self._name} factory>"

class SGPR(Reg): pass
class VGPR(Reg): pass
class TTMP(Reg): pass
s: _RegFactory[SGPR] = _RegFactory(SGPR, "SGPR")
v: _RegFactory[VGPR] = _RegFactory(VGPR, "VGPR")
ttmp: _RegFactory[TTMP] = _RegFactory(TTMP, "TTMP")

# Field type markers (runtime classes for validation)
class _SSrc: pass
class _Src: pass
class _Imm: pass
class _SImm: pass
class _VDSTYEnc: pass  # VOPD vdsty: encoded = actual >> 1, actual = (encoded << 1) | ((vdstx & 1) ^ 1)
class _SGPRField: pass
class _VGPRField: pass

# Type aliases for annotations - tells mypy it's a BitField while preserving marker info
SSrc = Annotated[BitField, _SSrc]
Src = Annotated[BitField, _Src]
Imm = Annotated[BitField, _Imm]
SImm = Annotated[BitField, _SImm]
VDSTYEnc = Annotated[BitField, _VDSTYEnc]
SGPRField = Annotated[BitField, _SGPRField]
VGPRField = Annotated[BitField, _VGPRField]
class RawImm:
  def __init__(self, val: int): self.val = val
  def __repr__(self): return f"RawImm({self.val})"
  def __eq__(self, other): return isinstance(other, RawImm) and self.val == other.val

def unwrap(val) -> int:
  return val.val if isinstance(val, RawImm) else val.value if hasattr(val, 'value') else val.idx if hasattr(val, 'idx') else val

# Encoding helpers
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
SRC_FIELDS = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'soffset', 'srcx0', 'srcy0'}
RAW_FIELDS = {'vdata', 'vdst', 'vaddr', 'addr', 'data', 'data0', 'data1', 'sdst', 'sdata'}

def _encode_reg(val) -> int:
  if isinstance(val, TTMP): return 108 + val.idx
  return val.idx | (0x80 if val.hi else 0)

def encode_src(val) -> int:
  if isinstance(val, VGPR): return 256 + _encode_reg(val)
  if isinstance(val, Reg): return _encode_reg(val)
  if hasattr(val, 'value'): return val.value
  if isinstance(val, float): return 128 if val == 0.0 else FLOAT_ENC.get(val, 255)
  return 128 + val if isinstance(val, int) and 0 <= val <= 64 else 192 + (-val) if isinstance(val, int) and -16 <= val <= -1 else 255

# Instruction base class
class Inst:
  _fields: dict[str, BitField]
  _encoding: tuple[BitField, int] | None = None
  _defaults: dict[str, int] = {}
  _values: dict[str, int | RawImm]
  _words: int  # size in 32-bit words, set by decode_program
  _literal: int | None

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {n: v[0] if isinstance(v, tuple) else v for n, v in cls.__dict__.items() if isinstance(v, BitField) or (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], BitField))}
    if 'encoding' in cls._fields and isinstance(cls.__dict__.get('encoding'), tuple): cls._encoding = cls.__dict__['encoding']

  def __init__(self, *args, literal: int | None = None, **kwargs):
    self._values, self._literal = dict(self._defaults), literal
    self._values.update(zip([n for n in self._fields if n != 'encoding'], args))
    self._values.update(kwargs)
    # Type check and encode values
    for name, val in list(self._values.items()):
      if name == 'encoding': continue
      # For RawImm, only process RAW_FIELDS to unwrap to int
      if isinstance(val, RawImm):
        if name in RAW_FIELDS: self._values[name] = val.val
        continue
      field = self._fields.get(name)
      marker = field.marker if field else None
      # Type validation
      if marker is _SGPRField:
        if isinstance(val, VGPR): raise TypeError(f"field '{name}' requires SGPR, got VGPR")
        if not isinstance(val, (SGPR, TTMP, int, RawImm)): raise TypeError(f"field '{name}' requires SGPR, got {type(val).__name__}")
      if marker is _VGPRField:
        if not isinstance(val, VGPR): raise TypeError(f"field '{name}' requires VGPR, got {type(val).__name__}")
      if marker is _SSrc and isinstance(val, VGPR): raise TypeError(f"field '{name}' requires scalar source, got VGPR")
      # Encode source fields as RawImm for consistent disassembly
      if name in SRC_FIELDS:
        encoded = encode_src(val)
        self._values[name] = RawImm(encoded)
        # Track literal value if needed (encoded as 255)
        if encoded == 255 and self._literal is None and isinstance(val, int) and not isinstance(val, IntEnum):
          self._literal = val
      # Encode raw register fields for consistent repr
      elif name in RAW_FIELDS:
        if isinstance(val, Reg): self._values[name] = _encode_reg(val)
        elif hasattr(val, 'value'): self._values[name] = val.value  # IntEnum like SrcEnum.NULL
      # Encode sbase (divided by 2) and srsrc/ssamp (divided by 4)
      elif name == 'sbase' and isinstance(val, Reg):
        self._values[name] = val.idx // 2
      elif name in {'srsrc', 'ssamp'} and isinstance(val, Reg):
        self._values[name] = val.idx // 4
      # VOPD vdsty: encode as actual >> 1 (constraint: vdsty parity must be opposite of vdstx)
      elif marker is _VDSTYEnc and isinstance(val, VGPR):
        self._values[name] = val.idx >> 1

  def _encode_field(self, name: str, val) -> int:
    if isinstance(val, RawImm): return val.val
    if name in {'srsrc', 'ssamp'}: return val.idx // 4 if isinstance(val, Reg) else val
    if name == 'sbase': return val.idx // 2 if isinstance(val, Reg) else val
    if name in RAW_FIELDS: return _encode_reg(val) if isinstance(val, Reg) else val
    if isinstance(val, Reg) or name in SRC_FIELDS: return encode_src(val)
    return val.value if hasattr(val, 'value') else val

  def to_int(self) -> int:
    word = (self._encoding[1] & self._encoding[0].mask()) << self._encoding[0].lo if self._encoding else 0
    for n, bf in self._fields.items():
      if n != 'encoding' and n in self._values: word |= (self._encode_field(n, self._values[n]) & bf.mask()) << bf.lo
    return word

  def _get_literal(self) -> int | None:
    for n in SRC_FIELDS:
      if n in self._values and not isinstance(v := self._values[n], RawImm) and isinstance(v, int) and not isinstance(v, IntEnum) and not (0 <= v <= 64 or -16 <= v <= -1): return v
    return None

  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(self._size(), 'little')
    return result + (lit & 0xffffffff).to_bytes(4, 'little') if (lit := self._get_literal() or getattr(self, '_literal', None)) else result

  @classmethod
  def _size(cls) -> int: return 4 if issubclass(cls, Inst32) else 8
  def size(self) -> int: return self._size() + (4 if self._literal is not None else 0)

  @classmethod
  def from_int(cls, word: int):
    inst = object.__new__(cls)
    inst._values = {n: RawImm(v) if n in SRC_FIELDS else v for n, bf in cls._fields.items() if n != 'encoding' for v in [(word >> bf.lo) & bf.mask()]}
    inst._literal = None
    return inst

  @classmethod
  def from_bytes(cls, data: bytes):
    inst = cls.from_int(int.from_bytes(data[:cls._size()], 'little'))
    op_val = inst._values.get('op', 0)
    has_literal = cls.__name__ == 'VOP2' and op_val in (44, 45, 55, 56)
    has_literal = has_literal or (cls.__name__ == 'SOP2' and op_val in (69, 70))
    for n in SRC_FIELDS:
      if n in inst._values and isinstance(inst._values[n], RawImm) and inst._values[n].val == 255: has_literal = True
    if has_literal and len(data) >= cls._size() + 4: inst._literal = int.from_bytes(data[cls._size():cls._size()+4], 'little')
    return inst

  def __repr__(self):
    # Use _fields order and exclude fields that are 0/default (for consistent repr after roundtrip)
    def is_zero(v): return (isinstance(v, int) and v == 0) or (isinstance(v, VGPR) and v.idx == 0 and v.count == 1)
    items = [(k, self._values[k]) for k in self._fields if k in self._values and k != 'encoding'
             and not (is_zero(self._values[k]) and k not in {'op'})]
    lit = f", literal={hex(self._literal)}" if self._literal is not None else ""
    return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in items)}{lit})"

  def disasm(self) -> str:
    from extra.assembly.rdna3.asm import disasm
    return disasm(self)

class Inst32(Inst): pass
class Inst64(Inst): pass
