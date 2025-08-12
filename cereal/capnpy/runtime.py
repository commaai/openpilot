from __future__ import annotations
import math, struct
from dataclasses import dataclass, field as dc_field
from enum import Enum, auto
from typing import Dict, List, Optional, Any

class TypeKind(Enum):
  PRIM = auto()
  TEXT = auto()
  DATA = auto()
  LIST = auto()
  STRUCT = auto()

@dataclass
class PrimType: name: str
@dataclass
class TextType: ...
@dataclass
class DataType: ...
@dataclass
class StructRef: name: str
@dataclass
class ListOf:
  elem_kind: TypeKind
  elem_type: Any   # PrimType or StructType (resolved later)

PRIM_SIZES = {
  "Bool": (1, False),
  "Int8": (8, True),  "UInt8": (8, False),
  "Int16": (16, True),"UInt16": (16, False),
  "Int32": (32, True),"UInt32": (32, False),
  "Int64": (64, True),"UInt64": (64, False),
  "Float32": (32, None), "Float64": (64, None),
}

@dataclass
class Field:
  name: str
  kind: TypeKind
  typ: Any
  data_bit_off: Optional[int] = None
  bits: Optional[int] = None
  float_: Optional[bool] = None
  ptr_index: Optional[int] = None
  list_stride_bits: Optional[int] = None

@dataclass
class StructType:
  name: str
  fields: List[Field] = dc_field(default_factory=list)
  data_size_bytes: int = 0
  ptr_count: int = 0

  def compute_layout(self) -> None:
    data_bits = 0
    bool_cursor = 0
    ptr_idx = 0
    for f in self.fields:
      if f.kind == TypeKind.PRIM:
        b, signed_or_float = PRIM_SIZES[f.typ.name]
        if f.typ.name == "Bool":
          f.data_bit_off, f.bits, f.float_ = bool_cursor, 1, False
          bool_cursor += 1
          data_bits = max(data_bits, _align_up(bool_cursor, 8))
        else:
          data_bits = _align_up(max(data_bits, bool_cursor), b)
          f.data_bit_off, f.bits = data_bits, b
          f.float_ = (signed_or_float is None)
          data_bits += b
      else:
        f.ptr_index = ptr_idx
        ptr_idx += 1
        if f.kind == TypeKind.LIST and isinstance(f.typ, ListOf) and f.typ.elem_kind == TypeKind.PRIM:
          ebits, _ = PRIM_SIZES[f.typ.elem_type.name]
          f.list_stride_bits = _align_up(ebits, 8)
    self.data_size_bytes = int(_align_up(data_bits, 64) // 8)
    self.ptr_count = ptr_idx

  def from_bytes(self, buf: bytes, offset: int = 0) -> "Reader":
    return Reader(self, memoryview(buf), offset)

  def new_builder(self, data_bytes: Optional[int] = None, extra_ptrs: int = 0) -> "Builder":
    data_len = data_bytes if data_bytes is not None else self.data_size_bytes
    total = data_len + (self.ptr_count + extra_ptrs) * 8
    mv = memoryview(bytearray(total))
    return Builder(self, mv, 0)

@dataclass
class Schema:
  structs: Dict[str, StructType]
  def __getattr__(self, name: str) -> StructType:
    try: return self.structs[name]
    except KeyError: raise AttributeError(name)

def _align_up(v_bits: int, align_bits: int) -> int:
  return (v_bits + align_bits - 1) // align_bits * align_bits

def _signed30(x: int) -> int:
  return x - (1<<30) if (x & (1<<29)) else x

class Reader:
  __slots__ = ("_st","_mv","_base","_cache")
  def __init__(self, st: StructType, mv: memoryview, base: int):
    self._st, self._mv, self._base = st, mv, base
    self._cache: Dict[str, Any] = {}

  def __getattr__(self, name: str) -> Any:
    return self.get(name)

  def get(self, field_name: str) -> Any:
    c = self._cache
    if field_name in c: return c[field_name]
    f = _field(self._st, field_name)
    if f.kind == TypeKind.PRIM:
      val = _read_prim(self._mv, self._base*8 + (f.data_bit_off or 0), f.bits or 0, f.float_, f.typ.name)
    elif f.kind == TypeKind.TEXT or f.kind == TypeKind.DATA:
      ptr_off, word = _ptr_word(self._base, self._st, self._mv, f)
      _assert_list(word)
      base, count = _list_base_count(self._mv, ptr_off, word)
      data = bytes(self._mv[base: base+count])
      val = data[:-1].decode("utf-8", "strict") if f.kind == TypeKind.TEXT else data
    elif f.kind == TypeKind.LIST:
      ptr_off, word = _ptr_word(self._base, self._st, self._mv, f)
      _assert_list(word)
      if f.typ.elem_kind == TypeKind.PRIM:
        ebits, _ = PRIM_SIZES[f.typ.elem_type.name]
        base, count = _list_base_count(self._mv, ptr_off, word)
        val = _PrimListView(self._mv, base, count, ebits)
      elif f.typ.elem_kind == TypeKind.STRUCT:
        base = _list_base(self._mv, ptr_off, word)
        tag = int.from_bytes(self._mv[base:base+8], "little")
        count = tag & 0xffffffff
        data_words = (tag >> 32) & 0xffff
        ptrs = (tag >> 48) & 0xffff
        stride = data_words*8 + ptrs*8
        first = base + 8
        val = _StructListView(self._mv, first, count, f.typ.elem_type, stride, data_words*8, ptrs)
      else:
        raise NotImplementedError("List of non-prim/non-struct")
    elif f.kind == TypeKind.STRUCT:
      ptr_off, word = _ptr_word(self._base, self._st, self._mv, f)
      _assert_struct(word)
      data_bytes = ((word >> 32) & 0xffff) * 8
      ptrs = (word >> 48) & 0xffff
      base = ptr_off + 8 + _signed30((word >> 2) & ((1<<30)-1)) * 8
      val = Reader(f.typ, self._mv, base)
    else:
      raise NotImplementedError
    c[field_name] = val
    return val

class Builder:
  __slots__ = ("_st","_mv","_base")
  def __init__(self, st: StructType, mv: memoryview, base: int):
    self._st, self._mv, self._base = st, mv, base
  def set(self, field_name: str, value: Any) -> None:
    f = _field(self._st, field_name)
    if f.kind != TypeKind.PRIM:
      raise NotImplementedError("Only primitive setters in first pass")
    _write_prim(self._mv, self._base*8 + (f.data_bit_off or 0), f.bits or 0, f.float_, f.typ.name, value)
  def to_bytes(self) -> bytes:
    return bytes(self._mv)

# --- helpers ---
def _field(st: StructType, name: str) -> Field:
  for f in st.fields:
    if f.name == name: return f
  raise KeyError(name)

def _ptr_word(base: int, st: StructType, mv: memoryview, f: Field):
  off = base + st.data_size_bytes + 8*(f.ptr_index or 0)
  word = int.from_bytes(mv[off:off+8], "little")
  return off, word

def _assert_list(word: int):
  if (word & 3) != 1: raise ValueError("pointer is not a list")
def _assert_struct(word: int):
  if (word & 3) != 0: raise ValueError("pointer is not a struct")

def _list_base(mv: memoryview, ptr_off: int, word: int) -> int:
  return ptr_off + 8 + _signed30((word >> 2) & ((1<<30)-1)) * 8
def _list_base_count(mv: memoryview, ptr_off: int, word: int):
  base = _list_base(mv, ptr_off, word)
  count = (word >> 35) & ((1<<29)-1)
  return base, count

def _read_prim(mv: memoryview, bit_off: int, nbits: int, is_float: Optional[bool], prim_name: str):
  if is_float:
    off = bit_off // 8
    return struct.unpack_from("<f" if nbits==32 else "<d", mv, off)[0]
  if nbits == 1:
    return bool(_read_bits(mv, bit_off, 1, False))
  signed = prim_name.startswith("Int")
  return _read_bits(mv, bit_off, nbits, signed)

def _write_prim(mv: memoryview, bit_off: int, nbits: int, is_float: Optional[bool], prim_name: str, value: Any):
  if is_float:
    off = bit_off // 8
    struct.pack_into("<f" if nbits==32 else "<d", mv, off, float(value))
    return
  if nbits == 1:
    _write_bits(mv, bit_off, 1, 1 if value else 0); return
  signed = prim_name.startswith("Int")
  _write_bits(mv, bit_off, nbits, int(value) & ((1<<nbits)-1))

def _read_bits(mv: memoryview, bit_off: int, nbits: int, signed: bool) -> int:
  byte_off = bit_off // 8
  shift = bit_off % 8
  nbytes = math.ceil((shift + nbits) / 8)
  chunk = int.from_bytes(mv[byte_off: byte_off+nbytes], "little", signed=False)
  mask = (1 << nbits) - 1
  val = (chunk >> shift) & mask
  if signed:
    sign = 1 << (nbits - 1)
    if val & sign: val -= (1 << nbits)
  return val

def _write_bits(mv: memoryview, bit_off: int, nbits: int, value: int):
  byte_off = bit_off // 8
  shift = bit_off % 8
  nbytes = math.ceil((shift + nbits) / 8)
  cur = int.from_bytes(mv[byte_off: byte_off+nbytes], "little", signed=False)
  mask = ((1 << nbits) - 1) << shift
  cur = (cur & ~mask) | ((value << shift) & mask)
  mv[byte_off: byte_off+nbytes] = cur.to_bytes(nbytes, "little", signed=False)

class _PrimListView:
  __slots__ = ("_mv","_base","_count","_bits")
  def __init__(self, mv, base, count, bits):
    self._mv, self._base, self._count, self._bits = mv, base, count, bits
  def __len__(self): return self._count
  def __getitem__(self, i):
    if not (0 <= i < self._count): raise IndexError
    off = self._base + i * (self._bits // 8)
    b = self._bits
    if b == 8:  return self._mv[off]
    if b == 16: return int.from_bytes(self._mv[off:off+2], "little", signed=False)
    if b == 32: return struct.unpack_from("<I", self._mv, off)[0]
    if b == 64: return struct.unpack_from("<Q", self._mv, off)[0]
    raise NotImplementedError

class _StructListView:
  __slots__ = ("_mv","_base","_count","_st","_stride","_data_bytes","_ptrs")
  def __init__(self, mv, base, count, st: StructType, stride, data_bytes, ptrs):
    self._mv, self._base, self._count, self._st = mv, base, count, st
    self._stride, self._data_bytes, self._ptrs = stride, data_bytes, ptrs
  def __len__(self): return self._count
  def __getitem__(self, i):
    if not (0 <= i < self._count): raise IndexError
    return Reader(self._st, self._mv, self._base + i*self._stride)
