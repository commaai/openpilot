"""
Binary struct parsing DSL.

Defines a declarative schema for binary messages using dataclasses
and type annotations.
"""

import struct
from enum import Enum
from dataclasses import dataclass, is_dataclass
from typing import Annotated, Any, TypeVar, get_args, get_origin


class FieldType:
  """Base class for field type descriptors."""


@dataclass(frozen=True)
class IntType(FieldType):
  bits: int
  signed: bool
  big_endian: bool = False

@dataclass(frozen=True)
class FloatType(FieldType):
  bits: int

@dataclass(frozen=True)
class BitsType(FieldType):
  bits: int

@dataclass(frozen=True)
class BytesType(FieldType):
  size: int

@dataclass(frozen=True)
class ArrayType(FieldType):
  element_type: Any
  count_field: str

@dataclass(frozen=True)
class SwitchType(FieldType):
  selector: str
  cases: dict[Any, Any]
  default: Any = None

@dataclass(frozen=True)
class EnumType(FieldType):
  base_type: FieldType
  enum_cls: type[Enum]

@dataclass(frozen=True)
class ConstType(FieldType):
  base_type: FieldType
  expected: Any

@dataclass(frozen=True)
class SubstreamType(FieldType):
  length_field: str
  element_type: Any

# Common types - little endian
u8 = IntType(8, False)
u16 = IntType(16, False)
u32 = IntType(32, False)
s8 = IntType(8, True)
s16 = IntType(16, True)
s32 = IntType(32, True)
f32 = FloatType(32)
f64 = FloatType(64)
# Big endian variants
u16be = IntType(16, False, big_endian=True)
u32be = IntType(32, False, big_endian=True)
s16be = IntType(16, True, big_endian=True)
s32be = IntType(32, True, big_endian=True)


def bits(n: int) -> BitsType:
  """Create a bit-level field type."""
  return BitsType(n)

def bytes_field(size: int) -> BytesType:
  """Create a fixed-size bytes field."""
  return BytesType(size)

def array(element_type: Any, count_field: str) -> ArrayType:
  """Create an array/repeated field."""
  return ArrayType(element_type, count_field)

def switch(selector: str, cases: dict[Any, Any], default: Any = None) -> SwitchType:
  """Create a switch-on field."""
  return SwitchType(selector, cases, default)

def enum(base_type: Any, enum_cls: type[Enum]) -> EnumType:
  """Create an enum-wrapped field."""
  field_type = _field_type_from_spec(base_type)
  if field_type is None:
    raise TypeError(f"Unsupported field type: {base_type!r}")
  return EnumType(field_type, enum_cls)

def const(base_type: Any, expected: Any) -> ConstType:
  """Create a constant-value field."""
  field_type = _field_type_from_spec(base_type)
  if field_type is None:
    raise TypeError(f"Unsupported field type: {base_type!r}")
  return ConstType(field_type, expected)

def substream(length_field: str, element_type: Any) -> SubstreamType:
  """Parse a fixed-length substream using an inner schema."""
  return SubstreamType(length_field, element_type)


class BinaryReader:
  def __init__(self, data: bytes):
    self.data = data
    self.pos = 0
    self.bit_pos = 0  # 0-7, position within current byte

  def _require(self, n: int) -> None:
    if self.pos + n > len(self.data):
      raise EOFError("Unexpected end of data")

  def _read_struct(self, fmt: str):
    self._align_to_byte()
    size = struct.calcsize(fmt)
    self._require(size)
    value = struct.unpack_from(fmt, self.data, self.pos)[0]
    self.pos += size
    return value

  def read_bytes(self, n: int) -> bytes:
    self._align_to_byte()
    self._require(n)
    result = self.data[self.pos : self.pos + n]
    self.pos += n
    return result

  def read_bits_int_be(self, n: int) -> int:
    result = 0
    bits_remaining = n
    while bits_remaining > 0:
      if self.pos >= len(self.data):
        raise EOFError("Unexpected end of data while reading bits")
      bits_in_byte = 8 - self.bit_pos
      bits_to_read = min(bits_remaining, bits_in_byte)
      byte_val = self.data[self.pos]
      shift = bits_in_byte - bits_to_read
      mask = (1 << bits_to_read) - 1
      extracted = (byte_val >> shift) & mask
      result = (result << bits_to_read) | extracted
      self.bit_pos += bits_to_read
      bits_remaining -= bits_to_read
      if self.bit_pos >= 8:
        self.bit_pos = 0
        self.pos += 1
    return result

  def _align_to_byte(self) -> None:
    if self.bit_pos > 0:
      self.bit_pos = 0
      self.pos += 1


T = TypeVar('T', bound='BinaryStruct')


class BinaryStruct:
  """Base class for binary struct definitions."""

  def __init_subclass__(cls, **kwargs) -> None:
    super().__init_subclass__(**kwargs)
    if cls is BinaryStruct:
      return
    if not is_dataclass(cls):
      dataclass(init=False)(cls)
    fields = list(getattr(cls, '__annotations__', {}).items())
    cls.__binary_fields__ = fields  # type: ignore[attr-defined]

    @classmethod
    def _read(inner_cls, reader: BinaryReader):
      obj = inner_cls.__new__(inner_cls)
      for name, spec in inner_cls.__binary_fields__:
        value = _parse_field(spec, reader, obj)
        setattr(obj, name, value)
      return obj

    cls._read = _read  # type: ignore[attr-defined]

  @classmethod
  def from_bytes(cls: type[T], data: bytes) -> T:
    """Parse struct from bytes."""
    reader = BinaryReader(data)
    return cls._read(reader)

  @classmethod
  def _read(cls: type[T], reader: BinaryReader) -> T:
    """Override in subclasses to implement parsing."""
    raise NotImplementedError


def _resolve_path(obj: Any, path: str) -> Any:
  cur = obj
  for part in path.split('.'):
    cur = getattr(cur, part)
  return cur

def _unwrap_annotated(spec: Any) -> tuple[Any, ...]:
  if get_origin(spec) is Annotated:
    return get_args(spec)[1:]
  return ()

def _field_type_from_spec(spec: Any) -> FieldType | None:
  if isinstance(spec, FieldType):
    return spec
  for item in _unwrap_annotated(spec):
    if isinstance(item, FieldType):
      return item
  return None


def _int_format(field_type: IntType) -> str:
  if field_type.bits == 8:
    return 'b' if field_type.signed else 'B'
  endian = '>' if field_type.big_endian else '<'
  if field_type.bits == 16:
    code = 'h' if field_type.signed else 'H'
  elif field_type.bits == 32:
    code = 'i' if field_type.signed else 'I'
  else:
    raise ValueError(f"Unsupported integer size: {field_type.bits}")
  return f"{endian}{code}"

def _float_format(field_type: FloatType) -> str:
  if field_type.bits == 32:
    return '<f'
  if field_type.bits == 64:
    return '<d'
  raise ValueError(f"Unsupported float size: {field_type.bits}")

def _parse_field(spec: Any, reader: BinaryReader, obj: Any) -> Any:
  field_type = _field_type_from_spec(spec)
  if field_type is not None:
    spec = field_type
  if isinstance(spec, ConstType):
    value = _parse_field(spec.base_type, reader, obj)
    if value != spec.expected:
      raise ValueError(f"Invalid constant: expected {spec.expected!r}, got {value!r}")
    return value
  if isinstance(spec, EnumType):
    raw = _parse_field(spec.base_type, reader, obj)
    try:
      return spec.enum_cls(raw)
    except ValueError:
      return raw
  if isinstance(spec, SwitchType):
    key = _resolve_path(obj, spec.selector)
    target = spec.cases.get(key, spec.default)
    if target is None:
      return None
    return _parse_field(target, reader, obj)
  if isinstance(spec, ArrayType):
    count = _resolve_path(obj, spec.count_field)
    return [_parse_field(spec.element_type, reader, obj) for _ in range(int(count))]
  if isinstance(spec, SubstreamType):
    length = _resolve_path(obj, spec.length_field)
    data = reader.read_bytes(int(length))
    sub_reader = BinaryReader(data)
    return _parse_field(spec.element_type, sub_reader, obj)
  if isinstance(spec, IntType):
    return reader._read_struct(_int_format(spec))
  if isinstance(spec, FloatType):
    return reader._read_struct(_float_format(spec))
  if isinstance(spec, BitsType):
    value = reader.read_bits_int_be(spec.bits)
    return bool(value) if spec.bits == 1 else value
  if isinstance(spec, BytesType):
    return reader.read_bytes(spec.size)
  if isinstance(spec, type) and issubclass(spec, BinaryStruct):
    return spec._read(reader)
  raise TypeError(f"Unsupported field spec: {spec!r}")
