# cereal/capnpy/__init__.py
from __future__ import annotations
import io, re
from typing import Dict, List, Tuple
from .runtime import (
  Schema, StructType, Field, Reader, Builder,
  TypeKind, ListOf, TextType, DataType, PrimType, StructRef,
)

__all__ = ["load", "Schema", "Reader", "Builder", "remove_import_hook"]

def remove_import_hook():
  # compatibility shim for pycapnp callers
  pass

# Brace-aware block finders (catch nested structs/enums)
_OPEN_STRUCT = re.compile(r'struct\s+([A-Za-z_]\w*)\s*(?:@[0-9A-Za-zx]+)?\s*\{', re.S)
_OPEN_ENUM   = re.compile(r'enum\s+([A-Za-z_]\w*)\s*(?:@[0-9A-Za-zx]+)?\s*\{', re.S)

# Top-level fields: name @ N : Type [= default] ;
_FIELD_RE    = re.compile(r'([A-Za-z_]\w*)\s*@\s*(\d+)\s*:\s*([^;]+);')

# List<T> or List(T)
_LIST_RE     = re.compile(r'List\s*(?:<\s*([^>]+)\s*>|\(\s*([^)]+)\s*\))\s*$')

_PRIMS = {
  "Bool","Int8","Int16","Int32","Int64",
  "UInt8","UInt16","UInt32","UInt64",
  "Float32","Float64","Text","Data",
}

def _iter_blocks(txt: str, kind: str):
  pat = _OPEN_STRUCT if kind == "struct" else _OPEN_ENUM
  for m in pat.finditer(txt):
    name = m.group(1)
    i = m.end() - 1  # at '{'
    depth = 0
    end = None
    for j in range(i, len(txt)):
      ch = txt[j]
      if ch == '{':
        depth += 1
      elif ch == '}':
        depth -= 1
        if depth == 0:
          end = j
          break
    if end is None:
      continue
    yield name, txt[i+1:end]

def _strip_nested_braces(body: str) -> str:
  # Keep only characters at top brace depth (skip nested structs/groups/enums/union blocks)
  out = []
  depth = 0
  for ch in body:
    if ch == '{':
      depth += 1
      continue
    if ch == '}':
      depth -= 1
      continue
    if depth == 0:
      out.append(ch)
  return ''.join(out)

def _parse_type(s: str, enums: set[str]) -> Tuple[TypeKind, object]:
  s = s.strip()
  if s == "Text": return TypeKind.TEXT, TextType()
  if s == "Data": return TypeKind.DATA, DataType()

  m = _LIST_RE.match(s)
  if m:
    inner = (m.group(1) or m.group(2)).strip()
    k, t = _parse_type(inner, enums)
    return TypeKind.LIST, ListOf(k, t)

  last = s.split('.')[-1]  # allow dotted refs

  # primitives (bare or dotted-last)
  if s in _PRIMS or last in _PRIMS:
    prim = last if last in _PRIMS else s
    return TypeKind.PRIM, PrimType(prim)

  # enums map to UInt16 (accept dotted names)
  if s in enums or last in enums:
    return TypeKind.PRIM, PrimType("UInt16")

  # struct reference (resolved later; dotted allowed)
  return TypeKind.STRUCT, StructRef(s)

def _parse_schema_text(txt: str) -> Schema:
  # collect enum names (top-level & nested)
  enums: set[str] = {name for name, _ in _iter_blocks(txt, "enum")}

  # collect structs (top-level & nested)
  structs_raw: Dict[str, List[Field]] = {}
  for name, body in _iter_blocks(txt, "struct"):
    by_ord: Dict[int, Field] = {}
    # strip nested blocks so we only see top-level fields of this struct
    stripped = _strip_nested_braces(body)
    for fm in _FIELD_RE.finditer(stripped):
      fname, ord_s, ty_s = fm.group(1), fm.group(2), fm.group(3).strip()
      if '=' in ty_s:  # drop default values
        ty_s = ty_s.split('=')[0].strip()
      kind, typ = _parse_type(ty_s, enums)
      by_ord[int(ord_s)] = Field(name=fname, kind=kind, typ=typ)
    ordered = [by_ord[k] for k in sorted(by_ord.keys())] if by_ord else []
    structs_raw[name] = ordered

  # instantiate StructType shells, attach fields
  types: Dict[str, StructType] = {n: StructType(n) for n in structs_raw}
  for n, flist in structs_raw.items():
    types[n].fields = flist

  # resolve struct refs (support dotted names by taking last segment)
  for st in types.values():
    for f in st.fields:
      if f.kind == TypeKind.STRUCT and isinstance(f.typ, StructRef):
        ref = f.typ.name
        if ref not in types:
          last = ref.split('.')[-1]
          if last in types:
            f.typ = types[last]
          else:
            raise ValueError(f"unknown struct {ref}")
      if f.kind == TypeKind.LIST and isinstance(f.typ, ListOf) and f.typ.elem_kind == TypeKind.STRUCT and isinstance(f.typ.elem_type, StructRef):
        ref = f.typ.elem_type.name
        if ref not in types:
          last = ref.split('.')[-1]
          if last in types:
            f.typ.elem_type = types[last]
          else:
            raise ValueError(f"unknown struct {ref}")

  # compute simple layouts
  for st in types.values():
    st.compute_layout()

  return Schema(types)

def load(path: str) -> Schema:
  with io.open(path, "r", encoding="utf-8") as f:
    return _parse_schema_text(f.read())
