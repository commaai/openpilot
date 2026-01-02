#!/usr/bin/env python3
# generates autogen/__init__.py by parsing the AMD RDNA3.5 ISA PDF
import re, pdfplumber, pathlib
from tinygrad.helpers import fetch

PDF_URL = "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content"
FIELD_TYPES = {'SSRC0': 'SSrc', 'SSRC1': 'SSrc', 'SOFFSET': 'SSrc', 'SADDR': 'SSrc', 'SRC0': 'Src', 'SRC1': 'Src', 'SRC2': 'Src',
  'SDST': 'SGPRField', 'SBASE': 'SGPRField', 'SDATA': 'SGPRField', 'SRSRC': 'SGPRField', 'VDST': 'VGPRField', 'VSRC1': 'VGPRField', 'VDATA': 'VGPRField',
  'VADDR': 'VGPRField', 'ADDR': 'VGPRField', 'DATA': 'VGPRField', 'DATA0': 'VGPRField', 'DATA1': 'VGPRField', 'SIMM16': 'SImm', 'OFFSET': 'Imm',
  'OPX': 'VOPDOp', 'OPY': 'VOPDOp', 'SRCX0': 'Src', 'SRCY0': 'Src', 'VSRCX1': 'VGPRField', 'VSRCY1': 'VGPRField', 'VDSTX': 'VGPRField', 'VDSTY': 'VDSTYEnc'}
FIELD_ORDER = {
  'SOP2': ['op', 'sdst', 'ssrc0', 'ssrc1'], 'SOP1': ['op', 'sdst', 'ssrc0'], 'SOPC': ['op', 'ssrc0', 'ssrc1'],
  'SOPK': ['op', 'sdst', 'simm16'], 'SOPP': ['op', 'simm16'], 'VOP1': ['op', 'vdst', 'src0'], 'VOPC': ['op', 'src0', 'vsrc1'],
  'VOP2': ['op', 'vdst', 'src0', 'vsrc1'], 'VOP3SD': ['op', 'vdst', 'sdst', 'src0', 'src1', 'src2', 'clmp'],
  'SMEM': ['op', 'sdata', 'sbase', 'soffset', 'offset', 'glc', 'dlc'], 'DS': ['op', 'vdst', 'addr', 'data0', 'data1'],
  'VOP3': ['op', 'vdst', 'src0', 'src1', 'src2', 'omod', 'neg', 'abs', 'clmp', 'opsel'],
  'VOP3P': ['op', 'vdst', 'src0', 'src1', 'src2', 'neg', 'neg_hi', 'opsel', 'opsel_hi', 'clmp'],
  'FLAT': ['op', 'vdst', 'addr', 'data', 'saddr', 'offset', 'seg', 'dlc', 'glc', 'slc'],
  'MUBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MTBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MIMG': ['op', 'vdata', 'vaddr', 'srsrc', 'ssamp', 'dmask', 'dim', 'unrm', 'dlc', 'glc', 'slc'],
  'EXP': ['en', 'target', 'vsrc0', 'vsrc1', 'vsrc2', 'vsrc3', 'done', 'row'],
  'VINTERP': ['op', 'vdst', 'src0', 'src1', 'src2', 'waitexp', 'clmp', 'opsel', 'neg'],
  'VOPD': ['opx', 'opy', 'vdstx', 'vdsty', 'srcx0', 'vsrcx1', 'srcy0', 'vsrcy1'],
  'LDSDIR': ['op', 'vdst', 'attr', 'attr_chan', 'wait_va']}
SRC_EXTRAS = {233: 'DPP8', 234: 'DPP8FI', 250: 'DPP16', 251: 'VCCZ', 252: 'EXECZ', 254: 'LDS_DIRECT'}
FLOAT_MAP = {'0.5': 'POS_HALF', '-0.5': 'NEG_HALF', '1.0': 'POS_ONE', '-1.0': 'NEG_ONE', '2.0': 'POS_TWO', '-2.0': 'NEG_TWO',
  '4.0': 'POS_FOUR', '-4.0': 'NEG_FOUR', '1/(2*PI)': 'INV_2PI', '0': 'ZERO'}

def parse_bits(s: str) -> tuple[int, int] | None:
  return (int(m.group(1)), int(m.group(2) or m.group(1))) if (m := re.match(r'\[(\d+)(?::(\d+))?\]', s)) else None

def parse_fields_table(table: list, fmt: str, enums: set[str]) -> list[tuple]:
  fields = []
  for row in table[1:]:
    if not row or not row[0]: continue
    name, bits_str = row[0].split('\n')[0].strip(), (row[1] or '').split('\n')[0].strip()
    if not (bits := parse_bits(bits_str)): continue
    enc_val, hi, lo = None, bits[0], bits[1]
    if name == 'ENCODING' and row[2] and (m := re.search(r"'b([01_]+)", row[2])):
      enc_bits = m.group(1).replace('_', '')
      enc_val = int(enc_bits, 2)
      declared_width, actual_width = hi - lo + 1, len(enc_bits)
      if actual_width > declared_width: lo = hi - actual_width + 1
    ftype = f"{fmt}Op" if name == 'OP' and f"{fmt}Op" in enums else FIELD_TYPES.get(name.upper())
    fields.append((name, hi, lo, enc_val, ftype))
  return fields

def generate(output_path: pathlib.Path|str|None = None) -> dict:
  """Generate RDNA3.5 instruction definitions from the AMD ISA PDF. Returns dict with formats for testing."""
  pdf = pdfplumber.open(fetch(PDF_URL))
  pages = pdf.pages[150:200]
  page_texts = [p.extract_text() or '' for p in pages]
  page_tables = [[t.extract() for t in p.find_tables()] for p in pages]
  full_text = '\n'.join(page_texts)

  # parse SSRC encoding from first page with VCC_LO
  src_enum = dict(SRC_EXTRAS)
  for text in page_texts[:10]:
    if 'SSRC0' in text and 'VCC_LO' in text:
      for m in re.finditer(r'^(\d+)\s+(\S+)', text, re.M):
        val, name = int(m.group(1)), m.group(2).rstrip('.:')
        if name in FLOAT_MAP: src_enum[val] = FLOAT_MAP[name]
        elif re.match(r'^[A-Z][A-Z0-9_]*$', name): src_enum[val] = name
      break

  # parse opcode tables
  enums: dict[str, dict[int, str]] = {}
  for m in re.finditer(r'Table \d+\. (\w+) Opcodes(.*?)(?=Table \d+\.|\n\d+\.\d+\.\d+\.\s+\w+\s*\nDescription|$)', full_text, re.S):
    if ops := {int(x.group(1)): x.group(2) for x in re.finditer(r'(\d+)\s+([A-Z][A-Z0-9_]+)', m.group(2))}:
      enums[m.group(1) + "Op"] = ops
  if vopd_m := re.search(r'Table \d+\. VOPD Y-Opcodes\n(.*?)(?=Table \d+\.|15\.\d)', full_text, re.S):
    if ops := {int(x.group(1)): x.group(2) for x in re.finditer(r'(\d+)\s+(V_DUAL_\w+)', vopd_m.group(1))}:
      enums["VOPDOp"] = ops
  enum_names = set(enums.keys())

  def is_fields_table(t) -> bool: return t and len(t) > 1 and t[0] and 'Field' in str(t[0][0] or '')
  def has_encoding(fields) -> bool: return any(f[0] == 'ENCODING' for f in fields)
  def has_header_before_fields(text) -> bool:
    return (pos := text.find('Field Name')) != -1 and bool(re.search(r'\d+\.\d+\.\d+\.\s+\w+\s*\n', text[:pos]))

  # find format headers with their page indices
  format_headers = []  # (fmt_name, page_idx)
  for i, text in enumerate(page_texts):
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n?Description', text): format_headers.append((m.group(1), i, m.start()))
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n', text):
      if m.start() > len(text) - 200 and 'Description' not in text[m.end():] and i + 1 < len(page_texts):
        next_text = page_texts[i + 1].lstrip()
        if next_text.startswith('Description') or (next_text.startswith('"RDNA') and 'Description' in next_text[:200]):
          format_headers.append((m.group(1), i, m.start()))

  # parse instruction formats
  formats: dict[str, list] = {}
  for fmt_name, page_idx, header_pos in format_headers:
    if fmt_name in formats: continue
    text, tables = page_texts[page_idx], page_tables[page_idx]
    field_pos = text.find('Field Name', header_pos)

    # find fields table with ENCODING (same page or up to 2 pages ahead)
    fields = None
    for offset in range(3):
      if page_idx + offset >= len(pages): break
      if offset > 0 and has_header_before_fields(page_texts[page_idx + offset]): break
      for t in page_tables[page_idx + offset] if offset > 0 or field_pos > header_pos else []:
        if is_fields_table(t) and (f := parse_fields_table(t, fmt_name, enum_names)) and has_encoding(f):
          fields = f
          break
      if fields: break

    # for modifier formats (no ENCODING), accept first fields table on same page
    if not fields and field_pos > header_pos:
      for t in tables:
        if is_fields_table(t) and (f := parse_fields_table(t, fmt_name, enum_names)):
          fields = f
          break

    if not fields: continue
    field_names = {f[0] for f in fields}

    # check next pages for continuation fields (tables without ENCODING)
    for pg_offset in range(1, 3):
      if page_idx + pg_offset >= len(pages) or has_header_before_fields(page_texts[page_idx + pg_offset]): break
      for t in page_tables[page_idx + pg_offset]:
        if is_fields_table(t) and (extra := parse_fields_table(t, fmt_name, enum_names)) and not has_encoding(extra):
          for ef in extra:
            if ef[0] not in field_names:
              fields.append(ef)
              field_names.add(ef[0])
          break
    formats[fmt_name] = fields

  # fix known PDF errors (verified against LLVM test vectors)
  # SMEM: PDF says DLC=bit14, GLC=bit16 but actual encoding is DLC=bit13, GLC=bit14
  if 'SMEM' in formats:
    formats['SMEM'] = [(n, 13 if n == 'DLC' else 14 if n == 'GLC' else h, 13 if n == 'DLC' else 14 if n == 'GLC' else l, e, t)
                       for n, h, l, e, t in formats['SMEM']]

  # generate output
  def enum_lines(name, items):
    return [f"class {name}(IntEnum):"] + [f"  {n} = {v}" for v, n in sorted(items.items())] + [""]
  def field_key(f): return order.index(f[0].lower()) if f[0].lower() in order else 1000
  lines = ["# autogenerated from AMD RDNA3.5 ISA PDF by gen.py - do not edit", "from enum import IntEnum",
           "from typing import Annotated",
           "from extra.assembly.rdna3.lib import bits, BitField, Inst32, Inst64, SGPR, VGPR, TTMP as TTMP, s as s, v as v, ttmp as ttmp, SSrc, Src, SImm, Imm, VDSTYEnc, SGPRField, VGPRField",
           "import functools", ""]
  lines += enum_lines("SrcEnum", src_enum) + sum([enum_lines(n, ops) for n, ops in sorted(enums.items())], [])
  # Format-specific field defaults (verified against LLVM test vectors)
  format_defaults = {'VOP3P': {'opsel_hi': 3, 'opsel_hi2': 1}}
  lines.append("# instruction formats")
  for fmt_name, fields in sorted(formats.items()):
    base = "Inst64" if max(f[1] for f in fields) > 31 or fmt_name == 'VOP3SD' else "Inst32"
    order = FIELD_ORDER.get(fmt_name, [])
    lines.append(f"class {fmt_name}({base}):")
    if enc := next((f for f in fields if f[0] == 'ENCODING'), None):
      enc_str = f"bits[{enc[1]}:{enc[2]}] == 0b{enc[3]:b}" if enc[1] != enc[2] else f"bits[{enc[1]}] == {enc[3]}"
      lines.append(f"  encoding = {enc_str}")
    if defaults := format_defaults.get(fmt_name):
      lines.append(f"  _defaults = {defaults}")
    for name, hi, lo, _, ftype in sorted([f for f in fields if f[0] != 'ENCODING'], key=field_key):
      # Wrap IntEnum types (ending in Op) with Annotated[BitField, ...] for correct typing
      if ftype and ftype.endswith('Op'):
        ann = f":Annotated[BitField, {ftype}]"
      else:
        ann = f":{ftype}" if ftype else ""
      lines.append(f"  {name.lower()}{ann} = bits[{hi}]" if hi == lo else f"  {name.lower()}{ann} = bits[{hi}:{lo}]")
    lines.append("")
  lines.append("# instruction helpers")
  for cls_name, ops in sorted(enums.items()):
    fmt = cls_name[:-2]
    for op_val, name in sorted(ops.items()):
      seg = {"GLOBAL": ", seg=2", "SCRATCH": ", seg=2"}.get(fmt, "")
      tgt = {"GLOBAL": "FLAT, GLOBALOp", "SCRATCH": "FLAT, SCRATCHOp"}.get(fmt, f"{fmt}, {cls_name}")
      if fmt in formats or fmt in ("GLOBAL", "SCRATCH"):
        # VOP1/VOP2/VOPC get _e32 suffix, VOP3 promoted ops (< 512) get _e64 suffix
        if fmt in ("VOP1", "VOP2", "VOPC"):
          suffix = "_e32"
        elif fmt == "VOP3" and op_val < 512:
          suffix = "_e64"
        else:
          suffix = ""
        lines.append(f"{name.lower()}{suffix} = functools.partial({tgt}.{name}{seg})")
  # export SrcEnum values, but skip DPP8/DPP16 which conflict with class names
  skip_exports = {'DPP8', 'DPP16'}
  lines += [""] + [f"{name} = SrcEnum.{name}" for _, name in sorted(src_enum.items()) if name not in skip_exports] + ["OFF = NULL\n"]

  if output_path is not None: pathlib.Path(output_path).write_text('\n'.join(lines))
  return {"formats": formats, "enums": enums, "src_enum": src_enum}

if __name__ == "__main__":
  result = generate("extra/assembly/rdna3/autogen/__init__.py")
  print(f"generated SrcEnum ({len(result['src_enum'])}) + {len(result['enums'])} opcode enums + {len(result['formats'])} format classes")
