from __future__ import annotations
import io, re
from typing import Dict, List, Tuple
from .runtime import (
  Schema, StructType, Field, Reader, Builder,
  TypeKind, ListOf, TextType, DataType, PrimType, StructRef,
)

__all__ = ["load", "Schema", "Reader", "Builder", "remove_import_hook"]

def remove_import_hook():
  pass  # compatibility shim

_STRUCT_RE = re.compile(r"struct\s+([A-Za-z_]\w*)\s*\{(.*?)\}", re.S)
_FIELD_RE  = re.compile(r"([A-Za-z_]\w*)\s*@\s*(\d+)\s*:\s*([^;]+);")
# supports List<T> and List(T)
_LIST_RE   = re.compile(r"List\s*(?:<\s*([^>]+)\s*>|\(\s*([^)]+)\s*\))\s*$")

_PRIMS = {
  "Bool","Int8","Int16","Int32","Int64",
  "UInt8","UInt16","UInt32","UInt64",
  "Float32","Float64",
}

def _parse_type(s: str) -> Tuple[TypeKind, object]:
  s = s.strip()
  if s == "Text": return TypeKind.TEXT, TextType()
  if s == "Data": return TypeKind.DATA, DataType()
  m = _LIST_RE.match(s)
  if m:
    inner = (m.group(1) or m.group(2)).strip()
    k, t = _parse_type(inner)      # allow lists of prims or structs
    return TypeKind.LIST, ListOf(k, t)
  if s in _PRIMS:
    return TypeKind.PRIM, PrimType(s)
  # otherwise assume struct ref (resolved later)
  return TypeKind.STRUCT, StructRef(s)

def _parse_schema_text(txt: str) -> Schema:
  structs_raw: Dict[str, List[Field]] = {}
  for sm in _STRUCT_RE.finditer(txt):
    name, body = sm.group(1), sm.group(2)
    by_ord: Dict[int, Field] = {}
    for fm in _FIELD_RE.finditer(body):
      fname, ord_s, ty_s = fm.group(1), fm.group(2), fm.group(3)
      ord_i = int(ord_s)
      kind, t = _parse_type(ty_s)
      by_ord[ord_i] = Field(name=fname, kind=kind, typ=t)
    ordered = [by_ord[k] for k in sorted(by_ord.keys())] if by_ord else []
    structs_raw[name] = ordered

  types: Dict[str, StructType] = {n: StructType(n) for n in structs_raw.keys()}
  for n, flist in structs_raw.items():
    types[n].fields = flist

  # resolve struct refs
  for st in types.values():
    for f in st.fields:
      if f.kind == TypeKind.STRUCT and isinstance(f.typ, StructRef):
        refname = f.typ.name
        if refname not in types:
          raise ValueError(f"unknown struct {refname}")
        f.typ = types[refname]
      if f.kind == TypeKind.LIST and isinstance(f.typ, ListOf) and f.typ.elem_kind == TypeKind.STRUCT and isinstance(f.typ.elem_type, StructRef):
        refname = f.typ.elem_type.name
        if refname not in types:
          raise ValueError(f"unknown struct {refname}")
        f.typ.elem_type = types[refname]

  # compute simple layouts
  for st in types.values():
    st.compute_layout()

  return Schema(types)

def load(path: str) -> Schema:
  with io.open(path, "r", encoding="utf-8") as f:
    return _parse_schema_text(f.read())
