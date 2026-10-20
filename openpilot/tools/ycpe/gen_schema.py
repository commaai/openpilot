#!/usr/bin/env python3
"""Dumps the cereal Event schema layout (offsets, sizes, defaults, enum names) to schema.json for the JS reader."""
import json
import struct
from pathlib import Path

from openpilot.cereal import log

PRIMITIVE = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"}
# short type codes keep the file small: b i8 u8 ... f32 f64, t(ext) d(ata) e(num) s(truct) l(ist) g(roup) v(oid) p(ointer)
CODE = {"bool": "b", "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64", "uint8": "u8", "uint16": "u16", "uint32": "u32",
        "uint64": "u64", "float32": "f32", "float64": "f64", "text": "t", "data": "d", "enum": "e", "struct": "s", "list": "l",
        "group": "g", "void": "v", "anyPointer": "p", "interface": "p"}

# structs and enums are lists; types refer to them by index
structs, enums = [], []
struct_index, enum_index = {}, {}


def bits(kind, value):
  # capnp stores scalars XORed with their default; emit the default's raw bits as one integer (0 if none)
  fmt = {"float32": "<f", "float64": "<d"}.get(kind)
  if fmt:
    return int.from_bytes(struct.pack(fmt, value), "little")
  return int(value)


def type_desc(t, schema):
  kind = t.which()
  if kind == "enum":
    return ["e", add_enum(schema)]
  if kind == "struct":
    return ["s", add_struct(schema)]
  if kind == "list":
    elem = t.list.elementType
    sub = schema.elementType if elem.which() in ("struct", "enum", "list") else None
    return ["l", type_desc(elem, sub)]
  return [CODE[kind]]


def add_enum(schema):
  key = schema.node.id
  if key not in enum_index:
    enum_index[key] = len(enums)
    enums.append([e.name for e in schema.node.enum.enumerants])
  return enum_index[key]


def add_struct(schema):
  key = schema.node.id
  if key in struct_index:
    return struct_index[key]
  node = schema.node
  entry = {
    "name": node.displayName.split(":")[-1],
    "data": node.struct.dataWordCount,
    "ptrs": node.struct.pointerCount,
    "disc": node.struct.discriminantOffset if node.struct.discriminantCount else -1,
    "fields": [],
  }
  struct_index[key] = len(structs)
  structs.append(entry)
  for name, f in schema.fields.items():
    p = f.proto
    disc = p.discriminantValue if p.discriminantValue != 0xffff else -1
    if p.which() == "group":
      entry["fields"].append([name, disc, "g", 0, add_struct(f.schema)])
      continue
    t = p.slot.type
    desc = type_desc(t, f.schema if t.which() in ("struct", "enum", "list") else None)
    field = [name, disc, desc[0], p.slot.offset, desc[1] if len(desc) > 1 else None]
    if t.which() in PRIMITIVE:
      d = getattr(p.slot.defaultValue, t.which())
      if d:
        field.append(bits(t.which(), d))
    elif desc[0] == "e" and p.slot.defaultValue.enum:
      field.append(p.slot.defaultValue.enum)
    entry["fields"].append(field)
  return struct_index[key]


if __name__ == "__main__":
  add_struct(log.Event.schema)  # Event is structs[0]
  out = {"structs": structs, "enums": enums}
  path = Path(__file__).parent / "data" / "schema.json"
  path.parent.mkdir(exist_ok=True)
  path.write_text(json.dumps(out, separators=(",", ":")))
  print(f"{path}: {len(structs)} structs, {len(enums)} enums, {path.stat().st_size // 1024} KB")
