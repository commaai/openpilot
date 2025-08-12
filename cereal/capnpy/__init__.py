# cereal/capnpy/__init__.py
from __future__ import annotations
import io, os, re
from typing import Dict, List, Tuple
from .runtime import (
  Schema, StructType, Field, Reader, Builder,
  TypeKind, ListOf, TextType, DataType, PrimType, StructRef,
)

__all__ = ["load", "Schema", "Reader", "Builder", "remove_import_hook", "KjException", "lib"]

def remove_import_hook():
  # compatibility shim for pycapnp callers
  pass

# pycapnp-compat exception + namespace alias
class KjException(Exception):
  pass

class _NS: pass
lib = _NS()
lib.capnp = _NS()
lib.capnp.KjException = KjException

# --- scanners ---
_OPEN_STRUCT = re.compile(r'struct\s+([A-Za-z_]\w*)\s*(?:@[0-9A-Za-zx]+)?\s*\{', re.S)
_OPEN_ENUM   = re.compile(r'enum\s+([A-Za-z_]\w*)\s*(?:@[0-9A-Za-zx]+)?\s*\{', re.S)
_FIELD_RE    = re.compile(r'([A-Za-z_]\w*)\s*@\s*(\d+)\s*:\s*([^;]+);')
_IMPORT_RE   = re.compile(r'using\s+([A-Za-z_]\w*)\s*=\s*import\s*"([^"]+)"\s*;')

_PRIMS = {
  "Bool","Int8","Int16","Int32","Int64",
  "UInt8","UInt16","UInt32","UInt64",
  "Float32","Float64","Text","Data",
}

def _iter_blocks_span(txt: str, kind: str):
  pat = _OPEN_STRUCT if kind == "struct" else _OPEN_ENUM
  for m in pat.finditer(txt):
    name = m.group(1)
    ob = m.end()-1
    depth = 0
    end = None
    for j in range(ob, len(txt)):
      ch = txt[j]
      if ch == '{': depth += 1
      elif ch == '}':
        depth -= 1
        if depth == 0:
          end = j
          break
    if end is None: continue
    yield name, txt[ob+1:end], (m.start(), end)

def _parse_enums(txt: str) -> set[str]:
  return {name for name, _, _ in _iter_blocks_span(txt, "enum")}

def _extract_fields_text(body: str) -> str:
  # Keep top-level fields; flatten union/group inner fields into top-level
  i, n, out = 0, len(body), []
  def skip_block(j):
    depth, k = 0, j
    while k < n:
      ch = body[k]
      if ch == '{': depth += 1
      elif ch == '}':
        depth -= 1
        if depth == 0: return k+1
      k += 1
    return n
  while i < n:
    if body.startswith("struct", i) and (i==0 or not body[i-1].isalnum()):
      j = body.find('{', i+6);
      if j == -1: break
      i = skip_block(j);
      continue
    if body.startswith("enum", i) and (i==0 or not body[i-1].isalnum()):
      j = body.find('{', i+4)
      if j == -1: break
      i = skip_block(j)
      continue
    if body.startswith("union", i) and (i==0 or not body[i-1].isalnum()):
      j = body.find('{', i+5)
      if j == -1: break
      k = skip_block(j)
      out.append(_extract_fields_text(body[j+1:k-1]))
      i = k
      continue
    if body.startswith("group", i) and (i==0 or not body[i-1].isalnum()):
      j = body.find('{', i+5)
      if j == -1: break
      k = skip_block(j)
      out.append(_extract_fields_text(body[j+1:k-1]))
      i = k
      continue
    out.append(body[i]); i += 1
  return ''.join(out)

def _parse_list_inner(s: str) -> str | None:
  s = s.strip()
  if not s.startswith("List"): return None
  i = len("List")
  while i < len(s) and s[i].isspace(): i += 1
  if i >= len(s) or s[i] not in "<(": return None
  open_ch = s[i]; close_ch = ">" if open_ch == "<" else ")"
  i += 1; depth = 1; start = i
  for j in range(i, len(s)):
    ch = s[j]
    if ch == open_ch: depth += 1
    elif ch == close_ch:
      depth -= 1
      if depth == 0:
        inner = s[start:j].strip()
        if s[j+1:].strip() == "": return inner
        return None
  return None

def _split_generic_args(args_s: str) -> List[str]:
  args, cur, da, dp = [], [], 0, 0
  for ch in args_s:
    if ch == ',' and da == 0 and dp == 0:
      args.append(''.join(cur).strip()); cur = []; continue
    if ch == '<': da += 1
    elif ch == '>': da -= 1
    elif ch == '(': dp += 1
    elif ch == ')': dp -= 1
    cur.append(ch)
  if cur: args.append(''.join(cur).strip())
  return args

def _parse_type_with_generics(s: str, enums: set[str],
                              templates: Dict[str, Tuple[List[str], str]],
                              param_map: Dict[str, Tuple[TypeKind, object]]):
  s = s.strip()
  if s == "Text": return TypeKind.TEXT, TextType()
  if s == "Data": return TypeKind.DATA, DataType()
  if s in param_map: return param_map[s]
  inner = _parse_list_inner(s)
  if inner is not None:
    k, t = _parse_type_with_generics(inner, enums, templates, param_map)
    return TypeKind.LIST, ListOf(k, t)
  last = s.split('.')[-1]
  if s in _PRIMS or last in _PRIMS:
    prim = last if last in _PRIMS else s
    return TypeKind.PRIM, PrimType(prim)
  if s in enums or last in enums:
    return TypeKind.PRIM, PrimType("UInt16")  # enums on wire
  m2 = re.match(r'^([A-Za-z_]\w*)\s*(?:<|\()(.*)(?:>|\))$', s)
  if m2 and m2.group(1) in templates:
    return TypeKind.STRUCT, StructRef(s)  # generic instantiation, resolve later
  return TypeKind.STRUCT, StructRef(s)

def _parse_with_imports(txt: str, base_dir: str, visited: Dict[str, Schema]) -> Schema:
  # imports first
  imported_types: Dict[str, StructType] = {}
  imported_enums: set[str] = set()
  for m in _IMPORT_RE.finditer(txt):
    rel = m.group(2)
    p = os.path.abspath(os.path.join(base_dir, rel))
    if not os.path.exists(p):
      continue  # ignore std includes (c++.capnp etc.)
    if p in visited:
      sch = visited[p]
    else:
      with io.open(p, "r", encoding="utf-8") as f2:
        sch = _parse_with_imports(f2.read(), os.path.dirname(p), visited)
      visited[p] = sch
    imported_types.update(sch.structs)
    imported_enums.update(getattr(sch, "_enums", set()))

  enums: set[str] = _parse_enums(txt).union(imported_enums)

  # collect generic templates in this file
  templates: Dict[str, Tuple[List[str], str]] = {}
  gen_spans: List[Tuple[int,int]] = []
  for m in re.finditer(r'struct\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{', txt):
    name = m.group(1)
    i = m.end()-1; depth = 0; end = None
    for j in range(i, len(txt)):
      ch = txt[j]
      if ch == '{': depth += 1
      elif ch == '}':
        depth -= 1
        if depth == 0:
          end = j; break
    if end is None: continue
    body = txt[i+1:end]
    params = [p.strip() for p in m.group(2).split(',') if p.strip()]
    templates[name] = (params, body)
    gen_spans.append((m.start(), end))
  def _inside_generic(pos: int) -> bool:
    return any(a<=pos<=b for (a,b) in gen_spans)

  # parse non-generic structs here
  structs_raw: Dict[str, List[Field]] = {}
  for name, body, (spos, _) in _iter_blocks_span(txt, "struct"):
    if _inside_generic(spos): continue
    by_ord: Dict[int, Field] = {}
    field_text = _extract_fields_text(body)
    for fm in _FIELD_RE.finditer(field_text):
      fname, ord_s, ty_s = fm.group(1), fm.group(2), fm.group(3).strip()
      if '=' in ty_s: ty_s = ty_s.split('=')[0].strip()
      kind, typ = _parse_type_with_generics(ty_s, enums, templates, {})
      by_ord[int(ord_s)] = Field(name=fname, kind=kind, typ=typ)
    structs_raw[name] = [by_ord[k] for k in sorted(by_ord.keys())] if by_ord else []

  # create local types and merge imported types into lookup
  types: Dict[str, StructType] = {n: StructType(n) for n in structs_raw}
  for n, flist in structs_raw.items(): types[n].fields = flist
  for k, v in imported_types.items():
    if k not in types: types[k] = v

  # instantiate generics on demand
  inst_cache: Dict[Tuple[str, Tuple[str,...]], StructType] = {}
  def ensure_struct_ref(refname: str) -> StructType:
    if refname in types: return types[refname]
    last = refname.split('.')[-1]
    if last in types: return types[last]
    m = re.match(r'^([A-Za-z_]\w*)\s*(?:<|\()\s*(.+)\s*(?:>|\))$', refname)
    if m:
      base, args_s = m.group(1), m.group(2)
      if base in templates:
        params, body = templates[base]
        args = _split_generic_args(args_s)
        arg_types = []
        for a in args:
          k, t = _parse_type_with_generics(a, enums, templates, {})
          if k == TypeKind.STRUCT and isinstance(t, StructRef):
            try: t = ensure_struct_ref(t.name)
            except Exception: pass
          arg_types.append((k, t, a.strip()))
        key = (base, tuple(a2 for (_,_,a2) in arg_types))
        if key in inst_cache: return inst_cache[key]
        inst_name = f"{base}<{','.join([a2 for (_,_,a2) in arg_types])}>"
        st = StructType(inst_name)
        inst_cache[key] = st

        param_map = {p: (k,t) for p,(k,t,_) in zip(params, arg_types)}
        by_ord: Dict[int, Field] = {}
        field_text = _extract_fields_text(body)
        for fm in _FIELD_RE.finditer(field_text):
          fname, ord_s, ty_s = fm.group(1), fm.group(2), fm.group(3).strip()
          if '=' in ty_s: ty_s = ty_s.split('=')[0].strip()
          k, t = _parse_type_with_generics(ty_s, enums, templates, param_map)
          if k == TypeKind.STRUCT and isinstance(t, StructRef):
            t = ensure_struct_ref(t.name)
          by_ord[int(ord_s)] = Field(name=fname, kind=k, typ=t)
        st.fields = [by_ord[k] for k in sorted(by_ord.keys())] if by_ord else []

        # nested structs inside template (e.g., Map.Entry)
        nested_types: Dict[str, StructType] = {}
        for nm, nb, _ in _iter_blocks_span(body, "struct"):
          nested_name = f"{inst_name}.{nm}"
          nst = StructType(nested_name)
          by_ord2: Dict[int, Field] = {}
          field_text2 = _extract_fields_text(nb)
          for fm in _FIELD_RE.finditer(field_text2):
            fname, ord_s, ty_s = fm.group(1), fm.group(2), fm.group(3).strip()
            if '=' in ty_s: ty_s = ty_s.split('=')[0].strip()
            k, t = _parse_type_with_generics(ty_s, enums, templates, param_map)
            if k == TypeKind.STRUCT and isinstance(t, StructRef):
              t = ensure_struct_ref(t.name)
            by_ord2[int(ord_s)] = Field(name=fname, kind=k, typ=t)
          nst.fields = [by_ord2[k] for k in sorted(by_ord2.keys())] if by_ord2 else []
          nst.compute_layout()
          nested_types[nm] = nst

        # resolve list<struct> refs against nested types
        for f in st.fields:
          if f.kind == TypeKind.LIST and isinstance(f.typ, ListOf) and f.typ.elem_kind == TypeKind.STRUCT and isinstance(f.typ.elem_type, StructRef):
            ref = f.typ.elem_type.name.split('.')[-1]
            if ref in nested_types: f.typ.elem_type = nested_types[ref]
            else: f.typ.elem_type = ensure_struct_ref(ref)

        st.compute_layout()
        types[inst_name] = st
        return st
    raise ValueError(f"unknown struct {refname}")

  # resolve refs + layouts
  for st in list(types.values()):
    for f in st.fields:
      if f.kind == TypeKind.STRUCT and isinstance(f.typ, StructRef):
        f.typ = ensure_struct_ref(f.typ.name)
      if f.kind == TypeKind.LIST and isinstance(f.typ, ListOf):
        if f.typ.elem_kind == TypeKind.STRUCT and isinstance(f.typ.elem_type, StructRef):
          f.typ.elem_type = ensure_struct_ref(f.typ.elem_type.name)

  for st in types.values(): st.compute_layout()

  sch = Schema(types)
  sch._enums = enums  # for dotted-enum handling in parents
  return sch

def load(path: str) -> Schema:
  path = os.path.abspath(path)
  with io.open(path, "r", encoding="utf-8") as f:
    txt = f.read()
  visited: Dict[str, Schema] = {}
  return _parse_with_imports(txt, os.path.dirname(path), visited)
