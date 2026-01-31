# AMD ISA code generator - generates enum.py, ins.py, operands.py, str_pcode.py
# Sources: XML from https://gpuopen.com/download/machine-readable-isa/latest/
#          PDF manuals from AMD documentation
import re, zlib, xml.etree.ElementTree as ET, zipfile
from tinygrad.helpers import fetch

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ARCHS = {
  "rdna3": {"xml": "amdgpu_isa_rdna3_5.xml", "pdf": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content"},
  "rdna4": {"xml": "amdgpu_isa_rdna4.xml", "pdf": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content"},
  "cdna": {"xml": "amdgpu_isa_cdna4.xml", "pdf": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf"},
}
XML_URL = "https://gpuopen.com/download/machine-readable-isa/latest/"
# Map XML encoding names to codebase names
NAME_MAP = {"VOP3_SDST_ENC": "VOP3SD", "VOP3_SDST_ENC_LIT": "VOP3SD_LIT", "VOP3_SDST_ENC_DPP16": "VOP3SD_DPP16",
            "VOP3_SDST_ENC_DPP8": "VOP3SD_DPP8", "VOPDXY": "VOPD", "VOPDXY_LIT": "VOPD_LIT", "VDS": "DS"}
# Instructions missing from XML but present in PDF
FIXES = {"rdna3": {"SOPK": {22: "S_SUBVECTOR_LOOP_BEGIN", 23: "S_SUBVECTOR_LOOP_END"}, "FLAT": {55: "FLAT_ATOMIC_CSUB_U32"}},
         "rdna4": {"SOP1": {80: "S_GET_BARRIER_STATE", 81: "S_BARRIER_INIT", 82: "S_BARRIER_JOIN"}, "SOPP": {9: "S_WAITCNT", 21: "S_BARRIER_LEAVE"}},
         "cdna": {"DS": {152: "DS_GWS_SEMA_RELEASE_ALL", 154: "DS_GWS_SEMA_V", 156: "DS_GWS_SEMA_P"},
                  "VOP3P": {44: "V_MFMA_LD_SCALE_B32", 62: "V_MFMA_F32_16X16X8_XF32", 63: "V_MFMA_F32_32X32X4_XF32"}}}
# Fields missing from XML but present in hardware (format: {arch: {encoding: [(name, hi, lo), ...]}})
FIELD_FIXES = {"cdna": {"VOP3P": [("opsel_hi2", 14, 14)]}}
# Encoding suffixes to strip (variants we don't generate separate classes for)
_ENC_SUFFIXES = ("_NSA1",)
# Encoding suffix to class suffix mapping (for variants we DO generate)
_ENC_SUFFIX_MAP = {"_INST_LITERAL": "_LIT", "_VOP_DPP16": "_DPP16", "_VOP_DPP": "_DPP16", "_VOP_DPP8": "_DPP8",
                   "_VOP_SDWA": "_SDWA", "_VOP_SDWA_SDST_ENC": "_SDWA_SDST", "_MFMA": "_MFMA"}
# Field name normalization
_FIELD_RENAMES = {"opsel_hi_2": "opsel_hi2", "op_sel_hi_2": "opsel_hi2", "op_sel": "opsel", "bound_ctrl": "bc",
                  "tgt": "target", "row_en": "row", "unorm": "unrm", "clamp": "clmp", "wait_exp": "waitexp",
                  "simm32": "literal", "dpp_ctrl": "dpp", "acc_cd": "acc_cd", "acc": "acc",
                  "dst_sel": "dst_sel", "dst_unused": "dst_unused", "src0_sel": "src0_sel", "src1_sel": "src1_sel"}
# Encoding variants to skip entirely (NSA is for MIMG graphics instructions)
_SKIP_ENCODINGS = ("NSA",)

# ═══════════════════════════════════════════════════════════════════════════════
# XML parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_enc(name: str) -> str:
  """Strip ENC_ prefix and normalize encoding suffixes."""
  name = name.removeprefix("ENC_")
  for sfx in _ENC_SUFFIXES: name = name.replace(sfx, "")
  # Process longer suffixes first to avoid partial matches (e.g., _VOP_DPP8 before _VOP_DPP)
  for old, new in sorted(_ENC_SUFFIX_MAP.items(), key=lambda x: -len(x[0])): name = name.replace(old, new)
  return name

def _norm_field(name: str) -> str:
  """Normalize field name to match expected names."""
  for old, new in _FIELD_RENAMES.items(): name = name.replace(old, new)
  return name

def _map_flat(enc_name: str, instr_name: str) -> str:
  """Map FLAT/GLOBAL/SCRATCH encoding to correct enum based on instruction prefix."""
  if enc_name in ("FLAT_GLBL", "FLAT_GLOBAL"): return "GLOBAL"
  if enc_name == "FLAT_SCRATCH": return "SCRATCH"
  if enc_name in ("FLAT", "VFLAT", "VGLOBAL", "VSCRATCH"):
    v = "V" if enc_name.startswith("V") else ""
    if instr_name.startswith("GLOBAL_"): return f"{v}GLOBAL"
    if instr_name.startswith("SCRATCH_"): return f"{v}SCRATCH"
    return f"{v}FLAT"
  return enc_name

# ═══════════════════════════════════════════════════════════════════════════════
# XML parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_xml(filename: str):
  root = ET.fromstring(zipfile.ZipFile(fetch(XML_URL)).read(filename))
  encodings, enums, types, fmts, op_types_set = {}, {}, {}, {}, set()
  # Extract HWREG and MSG enums from OperandTypes
  op_enum_map = {("OPR_HWREG", "ID"): "HWREG", ("OPR_SENDMSG_RTN", "MSG"): "MSG"}
  for ot in root.findall(".//OperandTypes/OperandType"):
    ot_name = ot.findtext("OperandTypeName")
    for field in ot.findall(".//Field"):
      if (enum_name := op_enum_map.get((ot_name, field.findtext("FieldName")))):
        enums[enum_name] = {int(pv.findtext("Value")): pv.findtext("Name").upper() for pv in field.findall(".//PredefinedValue")}
  # Extract DataFormats with BitCount
  for df in root.findall("ISA/DataFormats/DataFormat"):
    name, bits = df.findtext("DataFormatName"), df.findtext("BitCount")
    if name and bits: fmts[name] = int(bits)
  # Extract encoding definitions
  for enc in root.findall("ISA/Encodings/Encoding"):
    name = enc.findtext("EncodingName")
    is_base = name.startswith("ENC_") or name in ("VOP3_SDST_ENC", "VOPDXY")
    is_variant = any(sfx in name for sfx in _ENC_SUFFIX_MAP)
    if not is_base and not is_variant: continue
    if any(s in name for s in _SKIP_ENCODINGS): continue
    fields = [(_norm_field(f.findtext("FieldName").lower()), int(f.find("BitLayout/Range").findtext("BitOffset") or 0) + int(f.find("BitLayout/Range").findtext("BitCount") or 0) - 1,
               int(f.find("BitLayout/Range").findtext("BitOffset") or 0))
              for f in enc.findall(".//MicrocodeFormat/BitMap/Field") if f.find("BitLayout/Range") is not None]
    ident = (enc.findall("EncodingIdentifiers/EncodingIdentifier") or [None])[0]
    enc_field = next((f for f in fields if f[0] == "encoding"), None)
    # For multi-dword formats, encoding field may be in higher dword but identifier pattern is always in dword0; use % 32
    enc_bits = "".join(ident.text[len(ident.text)-1-b] for b in range(enc_field[1] % 32, (enc_field[2] % 32)-1, -1)) if ident is not None and enc_field else None
    base_name = _strip_enc(name)
    encodings[NAME_MAP.get(base_name, base_name)] = (fields, enc_bits)
  # Extract instruction opcodes and operand info
  # Track which encodings each opcode appears in (for detecting LIT-only ops)
  opcode_encs: dict[str, dict[int, set[str]]] = {}  # {base_fmt: {opcode: {enc_names}}}
  for instr in root.findall("ISA/Instructions/Instruction"):
    name = instr.findtext("InstructionName")
    for enc in instr.findall("InstructionEncodings/InstructionEncoding"):
      if enc.findtext("EncodingCondition") != "default": continue
      base, opcode = _map_flat(_strip_enc(enc.findtext("EncodingName")), name), int(enc.findtext("Opcode") or 0)
      enc_name = NAME_MAP.get(base, base)
      # Encoding variants use the same Op enum as the base format
      base_enum = enc_name
      for sfx in ("_SDWA_SDST", "_DPP16", "_DPP8", "_SDWA", "_LIT", "_MFMA"):
        base_enum = base_enum.replace(sfx, "")
      # Track which encodings this opcode appears in
      opcode_encs.setdefault(base_enum, {}).setdefault(opcode, set()).add(enc_name)
      # ADDTID instructions go in both FLAT and GLOBAL enums (pcode uses FLATOp for these)
      if "ADDTID" in name:
        if base == "GLOBAL": enums.setdefault("FLAT", {})[opcode] = name
        elif base == "VGLOBAL": enums.setdefault("VFLAT", {})[opcode] = name
      enums.setdefault(base_enum, {})[opcode] = name
      # Extract operand info
      op_info = {op.findtext("FieldName").lower(): (op.findtext("DataFormatName"), int(op.findtext("OperandSize") or 0), op.findtext("OperandType"))
                 for op in enc.findall("Operands/Operand") if op.findtext("FieldName")}
      for fmt, _, otype in op_info.values():
        if fmt and fmt not in fmts: fmts[fmt] = 0
        if otype: op_types_set.add(otype)
      if op_info: types[(name, base_enum)] = op_info
  # Find opcodes that only exist in a specific variant encoding (no base format version)
  suffix_only_ops: dict[str, dict[str, set[int]]] = {}  # {suffix: {base_fmt: {opcodes}}}
  for base_fmt, opcodes in opcode_encs.items():
    for opcode, encs in opcodes.items():
      suffix = next((s for s in _ENC_SUFFIX_MAP.values() if all(s in e for e in encs)), None)
      if suffix is not None: suffix_only_ops.setdefault(suffix, {}).setdefault(base_fmt, set()).add(opcode)
  return encodings, enums, types, fmts, op_types_set, suffix_only_ops

# ═══════════════════════════════════════════════════════════════════════════════
# PDF parsing
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(url: str) -> list[list[tuple[float, float, str, str]]]:
  """Extract positioned text from PDF. Returns list of text elements (x, y, text, font) per page."""
  data = fetch(url).read_bytes()
  # Parse xref table to locate objects
  xref: dict[int, int] = {}
  pos = int(re.search(rb'startxref\s+(\d+)', data).group(1)) + 4
  while data[pos:pos+7] != b'trailer':
    while data[pos:pos+1] in b' \r\n': pos += 1
    line_end = data.find(b'\n', pos)
    start_obj, count = map(int, data[pos:line_end].split()[:2])
    pos = line_end + 1
    for i in range(count):
      if data[pos+17:pos+18] == b'n' and (off := int(data[pos:pos+10])) > 0: xref[start_obj + i] = off
      pos += 20

  def get_stream(n: int) -> bytes:
    obj = data[xref[n]:data.find(b'endobj', xref[n])]
    raw = obj[obj.find(b'stream\n') + 7:obj.find(b'\nendstream')]
    return zlib.decompress(raw) if b'/FlateDecode' in obj else raw

  pages = []
  for n in sorted(xref):
    if b'/Type /Page' not in data[xref[n]:xref[n]+500]: continue
    if not (m := re.search(rb'/Contents (\d+) 0 R', data[xref[n]:xref[n]+500])): continue
    stream = get_stream(int(m.group(1))).decode('latin-1')
    elements, font = [], ''
    for bt in re.finditer(r'BT(.*?)ET', stream, re.S):
      x, y = 0.0, 0.0
      for m in re.finditer(r'(/F[\d.]+) [\d.]+ Tf|([\d.+-]+) ([\d.+-]+) Td|[\d.+-]+ [\d.+-]+ [\d.+-]+ [\d.+-]+ ([\d.+-]+) ([\d.+-]+) Tm|<([0-9A-Fa-f]+)>.*?Tj|\[([^\]]+)\] TJ', bt.group(1)):
        if m.group(1): font = m.group(1)
        elif m.group(2): x, y = x + float(m.group(2)), y + float(m.group(3))
        elif m.group(4): x, y = float(m.group(4)), float(m.group(5))
        elif m.group(6) and (t := bytes.fromhex(m.group(6)).decode('latin-1')).strip(): elements.append((x, y, t, font))
        elif m.group(7) and (t := ''.join(bytes.fromhex(h).decode('latin-1') for h in re.findall(r'<([0-9A-Fa-f]+)>', m.group(7)))).strip(): elements.append((x, y, t, font))
    pages.append(sorted(elements, key=lambda e: (-e[1], e[0])))
  return pages

def extract_pcode(pages: list[list[tuple[float, float, str, str]]], name_to_op: dict[str, int]) -> dict[tuple[str, int], str]:
  """Extract pseudocode for instructions. Returns {(name, opcode): pseudocode}."""
  # First pass: find all instruction headers across all pages
  all_instructions: list[tuple[int, float, str, int]] = []  # (page_idx, y, name, opcode)
  for page_idx, page in enumerate(pages):
    by_y: dict[int, list[tuple[float, str]]] = {}
    for x, y, t, _ in page:
      by_y.setdefault(round(y), []).append((x, t))
    for y, items in sorted(by_y.items(), reverse=True):
      left = [(x, t) for x, t in items if 55 < x < 65]
      right = [(x, t) for x, t in items if 535 < x < 550]
      if left and right and left[0][1] in name_to_op and right[0][1].isdigit():
        all_instructions.append((page_idx, y, left[0][1], int(right[0][1])))

  # Second pass: extract pseudocode between consecutive instructions
  pcode: dict[tuple[str, int], str] = {}
  for i, (page_idx, y, name, opcode) in enumerate(all_instructions):
    if i + 1 < len(all_instructions):
      next_page, next_y = all_instructions[i + 1][0], all_instructions[i + 1][1]
    else:
      next_page, next_y = page_idx, 0
    # Collect F6 text from current position to next instruction (pseudocode is at x ≈ 69)
    lines = []
    for p in range(page_idx, next_page + 1):
      start_y = y if p == page_idx else 800
      end_y = next_y if p == next_page else 0
      lines.extend((p, y2, t) for x, y2, t, f in pages[p] if f in ('/F6.0', '/F7.0') and end_y < y2 < start_y and 60 < x < 80)
    if lines:
      sorted_lines = sorted(lines, key=lambda x: (x[0], -x[1]))
      # Stop at large Y gaps (>30) - indicates section break
      filtered = [sorted_lines[0]]
      for j in range(1, len(sorted_lines)):
        prev_page, prev_y, _ = sorted_lines[j-1]
        curr_page, curr_y, _ = sorted_lines[j]
        if curr_page == prev_page and prev_y - curr_y > 30: break
        if curr_page != prev_page and prev_y > 60 and curr_y < 730: break
        filtered.append(sorted_lines[j])
      pcode_lines = [t.replace('Ê', '').strip() for _, _, t in filtered]
      if pcode_lines: pcode[(name, opcode)] = '\n'.join(pcode_lines)
  return pcode

# ═══════════════════════════════════════════════════════════════════════════════
# Code generation
# ═══════════════════════════════════════════════════════════════════════════════

def write_common(all_fmts, all_op_types, path):
  lines = ["# autogenerated from AMD ISA XML - do not edit", "from enum import Enum, auto", ""]
  lines.append("class ReprEnum(Enum):")
  lines.append('  """Enum with clean repr that roundtrips with eval()."""')
  lines.append('  def __repr__(self): return f"{type(self).__name__}.{self.name}"')
  lines.append("")
  lines.append("class Fmt(Enum):")
  for fmt in sorted(all_fmts.keys()): lines.append(f"  {fmt} = auto()")
  lines.append("")
  lines.append("FMT_BITS = {")
  for fmt, bits in sorted(all_fmts.items()): lines.append(f"  Fmt.{fmt}: {bits},")
  lines.append("}")
  lines.append("")
  lines.append("class OpType(Enum):")
  for ot in sorted(all_op_types): lines.append(f"  {ot} = auto()")
  with open(path, "w") as f: f.write("\n".join(lines))

def write_enum(enums, path):
  lines = ["# autogenerated from AMD ISA XML - do not edit", "from extra.assembly.amd.autogen.common import ReprEnum, Fmt, FMT_BITS, OpType  # noqa: F401", ""]
  for name, ops in sorted(enums.items()):
    if not ops: continue
    suffix = "_E32" if name in ("VOP1", "VOP2", "VOPC") else "_E64" if name == "VOP3" else ""
    lines.append(f"class {name}(ReprEnum):" if name in ("HWREG", "MSG") else f"class {name}Op(ReprEnum):")
    aliases = []
    for op, mem in sorted(ops.items()):
      msuf = suffix if name != "VOP3" or op < 512 else ""
      lines.append(f"  {mem}{msuf} = {op}")
      if msuf: aliases.append((mem, msuf))
    for mem, msuf in aliases: lines.append(f"  {mem} = {mem}{msuf}")
    lines.append("")
  with open(path, "w") as f: f.write("\n".join(lines))

def write_ins(encodings, enums, suffix_only_ops, types, arch, path):
  _VGPR_FIELDS = {"vdst", "vdstx", "vsrc0", "vsrc1", "vsrc2", "vsrc3", "vsrcx1", "vsrcy1", "vaddr", "vdata", "data", "data0", "data1", "addr", "vsrc"}
  _VARIANT_SUFFIXES = ("_LIT", "_DPP16", "_DPP8", "_SDWA_SDST", "_SDWA", "_MFMA")
  def get_base_fmt(fmt):
    for sfx in _VARIANT_SUFFIXES: fmt = fmt.replace(sfx, "")
    return fmt
  def field_def(name, hi, lo, fmt, enc_bits=None):
    bits = hi - lo + 1
    base_fmt = get_base_fmt(fmt)
    if name == "encoding" and enc_bits: return f"FixedBitField({hi}, {lo}, 0b{enc_bits})"
    if name == "op" and fmt not in ("DPP", "SDWA"): return f"EnumBitField({hi}, {lo}, {base_fmt}Op)"
    if name in ("opx", "opy"): return f"EnumBitField({hi}, {lo}, VOPDOp)"
    if name == "vdsty": return f"VDSTYField({hi}, {lo})"
    if name in _VGPR_FIELDS and bits == 8: return f"VGPRField({hi}, {lo})"
    if name == "sbase" and bits == 6: return f"SBaseField({hi}, {lo})"
    if name in ("srsrc", "ssamp") and bits == 5: return f"SRsrcField({hi}, {lo})"
    if name in ("sdst", "sdata") and bits == 7: return f"SGPRField({hi}, {lo})"
    if name in ("soffset", "saddr") and bits == 7: return f"SGPRField({hi}, {lo}, default=NULL)"
    if name.startswith("ssrc") and bits == 8: return f"SSrcField({hi}, {lo})"
    if name in ("saddr", "soffset") and bits == 8: return f"SSrcField({hi}, {lo}, default=NULL)"
    if name.startswith("src") and bits == 9: return f"SrcField({hi}, {lo})"
    # GLOBAL/SCRATCH: offset is 13-bit signed [12:0], FLAT: 12-bit unsigned (XML has 12-bit for all)
    if name == "offset" and base_fmt in ("GLOBAL", "SCRATCH"): return f"BitField(12, {lo})"
    if base_fmt == "VOP3P" and name == "opsel_hi": return f"BitField({hi}, {lo}, default=3)"
    if base_fmt == "VOP3P" and name == "opsel_hi2": return f"BitField({hi}, {lo}, default=1)"
    return f"BitField({hi}, {lo})"
  ORDER = ['encoding', 'op', 'opx', 'opy', 'vdst', 'vdstx', 'vdsty', 'sdst', 'vdata', 'sdata', 'addr', 'vaddr', 'data', 'data0', 'data1',
           'src0', 'srcx0', 'srcy0', 'vsrc0', 'ssrc0', 'src1', 'vsrc1', 'vsrcx1', 'vsrcy1', 'ssrc1', 'src2', 'vsrc2', 'src3', 'vsrc3',
           'saddr', 'sbase', 'srsrc', 'ssamp', 'soffset', 'offset', 'simm16', 'literal', 'en', 'target', 'attr', 'attr_chan',
           'omod', 'neg', 'neg_hi', 'abs', 'clmp', 'opsel', 'opsel_hi', 'waitexp', 'wait_va',
           'dmask', 'dim', 'seg', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe', 'unrm', 'done', 'row',
           'dpp', 'fi', 'bc', 'row_mask', 'bank_mask', 'src0_neg', 'src0_abs', 'src1_neg', 'src1_abs',
           'cbsz', 'abid', 'acc_cd', 'acc', 'blgp', 'lane_sel_0', 'lane_sel_1', 'lane_sel_2', 'lane_sel_3',
           'lane_sel_4', 'lane_sel_5', 'lane_sel_6', 'lane_sel_7', 'dst_sel', 'dst_unused', 'src0_sel', 'src1_sel']
  sort_fields = lambda fields: sorted(fields, key=lambda f: (ORDER.index(f[0]) if f[0] in ORDER else 999, f[2]))

  # Separate base encodings from variants
  base_encodings, variant_encodings = {}, {}
  for enc_name, data in encodings.items():
    base = get_base_fmt(enc_name)
    if base == enc_name: base_encodings[enc_name] = data
    else: variant_encodings[enc_name] = data

  # Build sets of ops by their vdst type from operand metadata
  sdst_opcodes = {}  # ops where vdst is OPR_SREG (writes to SGPR)
  for fmt, ops in enums.items():
    for op, name in ops.items():
      op_types = types.get((name, fmt), {})
      vdst_type = op_types.get("vdst", (None, None, None))[2]
      if vdst_type == "OPR_SREG": sdst_opcodes.setdefault(fmt, set()).add(op)

  lines = ["# autogenerated from AMD ISA XML - do not edit", "# ruff: noqa: F401,F403",
           "from extra.assembly.amd.dsl import *", f"from extra.assembly.amd.autogen.{arch}.enum import *", "import functools", ""]

  def fmt_allowed(op_enum: str, ops: set[int]) -> str:
    """Format allowed ops as {EnumName.MEMBER, ...}."""
    names = [f"{op_enum}.{enums[op_enum.removesuffix('Op')][op]}" for op in sorted(ops)]
    return "{" + ", ".join(names) + "}"

  # Generate base classes first
  for enc_name, (fields, enc_bits) in sorted(base_encodings.items()):
    all_ops = set(enums.get(enc_name, {}).keys())
    # Get suffix-only ops for this format (these can't be used in base class)
    base_suffix_ops = set().union(*(d.get(enc_name, set()) for d in suffix_only_ops.values()))
    # Exclude SDST ops from base class (they need VOP1_SDST/VOP3_SDST/VOP3B)
    base_allowed = all_ops - base_suffix_ops - sdst_opcodes.get(enc_name, set())
    # RDNA3 FLAT/GLOBAL/SCRATCH share encoding bits, differentiated by seg field
    # RDNA4 VFLAT/VGLOBAL/VSCRATCH have distinct encoding bits, no seg field needed
    has_seg_field = any(fn == "seg" for fn, _, _ in fields)
    if enc_name in ("FLAT", "VFLAT") and has_seg_field:
      prefix = "V" if enc_name == "VFLAT" else ""
      for cls, seg, op_enum in [(f"{prefix}FLAT", 0, f"{prefix}FLATOp"), (f"{prefix}GLOBAL", 2, f"{prefix}GLOBALOp"), (f"{prefix}SCRATCH", 1, f"{prefix}SCRATCHOp")]:
        cls_ops = set(enums.get(cls, {}).keys())
        lines.append(f"class {cls}(Inst):")
        for fn, hi, lo in sort_fields(fields):
          if fn == "seg": lines.append(f"  seg = FixedBitField({hi}, {lo}, {seg})")
          elif fn == "op": lines.append(f"  op = EnumBitField({hi}, {lo}, {op_enum}, {fmt_allowed(op_enum, cls_ops)})")
          else: lines.append(f"  {fn} = {field_def(fn, hi, lo, cls, enc_bits)}")
        lines.append("")
    elif enc_name not in ("FLAT_GLOBAL", "FLAT_SCRATCH", "FLAT_GLBL", "DPP", "SDWA"):
      lines.append(f"class {enc_name}(Inst):")
      for fn, hi, lo in sort_fields(fields):
        if fn == "op":
          base_fmt = get_base_fmt(enc_name)
          lines.append(f"  op = EnumBitField({hi}, {lo}, {base_fmt}Op, {fmt_allowed(f'{base_fmt}Op', base_allowed)})")
        else:
          lines.append(f"  {fn} = {field_def(fn, hi, lo, enc_name, enc_bits if fn == 'encoding' else None)}")
      lines.append("")

  # Generate variant classes that inherit from base (only add extra fields)
  for enc_name, (fields, enc_bits) in sorted(variant_encodings.items()):
    base = get_base_fmt(enc_name)
    if base not in base_encodings: continue  # skip if no base class
    base_fields = {f[0] for f in base_encodings[base][0]}
    extra_fields = [(fn, hi, lo) for fn, hi, lo in fields if fn not in base_fields]
    # Check if this is a suffix-only variant
    variant_suffix = next((sfx for sfx in _VARIANT_SUFFIXES if enc_name.endswith(sfx)), None)
    is_suffix_variant = variant_suffix in suffix_only_ops
    all_ops = set(enums.get(base, {}).keys())
    if extra_fields or is_suffix_variant:
      lines.append(f"class {enc_name}({base}):")
      op_field = next((f for f in base_encodings[base][0] if f[0] == "op"), None)
      # _LIT classes: override op to allow all opcodes (base excludes lit-only ops)
      # other classes override op to only suffix-only opcodes
      if op_field and is_suffix_variant:
        _, hi, lo = op_field
        allowed_ops = all_ops if variant_suffix == "_LIT" else suffix_only_ops[variant_suffix][base]
        lines.append(f"  op = EnumBitField({hi}, {lo}, {base}Op, {fmt_allowed(f'{base}Op', allowed_ops)})")
      for fn, hi, lo in sort_fields(extra_fields):
        lines.append(f"  {fn} = {field_def(fn, hi, lo, enc_name)}")
      lines.append("")

  # SDST variants (special case - redefine vdst field type, restrict to SDST ops)
  for base, field_hi, field_lo in [("VOP1", 24, 17), ("VOP3", 7, 0)]:
    if base not in base_encodings: continue
    sdst_ops = sdst_opcodes.get(base, set())
    if not sdst_ops: continue
    # For VOP3, all ops < 256 (compare/cmpx ops) use SDST encoding
    all_base_ops = set(enums.get(base, {}).keys())
    if base == "VOP3": sdst_ops = sdst_ops | {op for op in all_base_ops if op < 256}
    op_field = next((f for f in base_encodings[base][0] if f[0] == "op"), None)
    lines.append(f"class {base}_SDST({base}):")
    if op_field:
      _, hi, lo = op_field
      lines.append(f"  op = EnumBitField({hi}, {lo}, {base}Op, {fmt_allowed(f'{base}Op', sdst_ops)})")
    lines.append(f"  vdst = SSrcField({field_hi}, {field_lo})")
    lines.append("")
    # SDST_LIT class (for literals with SDST destination) - same ops, just adds literal field
    lit_enc = variant_encodings.get(f"{base}_LIT")
    if lit_enc:
      lit_field = next((f for f in lit_enc[0] if f[0] == "literal"), None)
      if lit_field:
        lines.append(f"class {base}_SDST_LIT({base}_SDST):")
        lines.append(f"  literal = BitField({lit_field[1]}, {lit_field[2]})")
        lines.append("")

  # Instruction helpers
  lines.append("# instruction helpers")
  for fmt, ops in sorted(enums.items()):
    if fmt not in base_encodings and fmt not in ("GLOBAL", "SCRATCH", "VGLOBAL", "VSCRATCH"): continue
    suffix = "_E32" if fmt in ("VOP1", "VOP2", "VOPC") else "_E64" if fmt == "VOP3" else ""
    op_to_suffix = {op:suffix for suffix,ops in suffix_only_ops.items() for op in ops.get(fmt, set())}
    fmt_sdst_ops = sdst_opcodes.get(fmt, set())
    for op, name in sorted(ops.items()):
      msuf = suffix if fmt != "VOP3" or op < 512 else ""
      # Determine class: SDST variants, suffix-specific variants (e.g., _MFMA, _LIT), or base
      if fmt == "VOP1" and op in fmt_sdst_ops: cls = "VOP1_SDST"
      elif fmt == "VOP3" and (op in fmt_sdst_ops or op < 256): cls = "VOP3_SDST"
      elif op_to_suffix.get(op): cls = f"{fmt}{op_to_suffix[op]}"
      else: cls = fmt
      lines.append(f"{name.lower()}{msuf.lower()} = functools.partial({cls}, {fmt}Op.{name}{msuf})")
  with open(path, "w") as f: f.write("\n".join(lines))

def write_operands(types, enums, arch, path):
  valid = {(name, fmt) for fmt, ops in enums.items() for name in ops.values()}
  lines = ["# autogenerated from AMD ISA XML - do not edit",
           "from extra.assembly.amd.autogen.common import Fmt, OpType",
           f"from extra.assembly.amd.autogen.{arch}.enum import *", ""]
  lines.append("# instruction operand info: {Op: {field: (Fmt, size_bits, OpType)}}")
  lines.append("OPERANDS = {")
  def fmt_val(v):
    fmt, size, otype = v
    return f"({f'Fmt.{fmt}' if fmt else 'None'}, {size}, {f'OpType.{otype}' if otype else 'None'})"
  for (name, enc_base), fields in sorted(types.items()):
    if (name, enc_base) not in valid: continue
    fstr = ", ".join(f'"{k}": {fmt_val(v)}' for k, v in sorted(fields.items()))
    lines.append(f'  {enc_base}Op.{name}: {{{fstr}}},')
  lines.append("}")
  with open(path, "w") as f: f.write("\n".join(lines))

def write_pcode(pcode: dict[tuple[str, int], str], enums: dict[str, dict[int, str]], arch: str, path: str):
  """Write str_pcode.py file from extracted pseudocode."""
  entries: list[tuple[str, str, int, str]] = []
  for fmt_name, ops in enums.items():
    member_suffix = "_E32" if fmt_name in ("VOP1", "VOP2", "VOPC") else "_E64" if fmt_name == "VOP3" else ""
    for opcode, name in ops.items():
      if (name, opcode) in pcode:
        msuf = member_suffix if fmt_name != "VOP3" or opcode < 512 else ""
        entries.append((f"{fmt_name}Op", f"{name}{msuf}", opcode, pcode[(name, opcode)]))
  enum_names = sorted(set(e[0] for e in entries))
  lines = ["# autogenerated from AMD ISA PDF - do not edit", "# ruff: noqa: E501",
           f"from extra.assembly.amd.autogen.{arch}.enum import {', '.join(enum_names)}", "", "PCODE = {"]
  for enum_name, name, opcode, code in sorted(entries, key=lambda x: (x[0], x[2])):
    lines.append(f"  {enum_name}.{name}: {code!r},")
  lines.append("}")
  with open(path, "w") as f: f.write("\n".join(lines))

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
  import pathlib
  all_fmts, all_op_types, arch_data = {}, set(), {}
  # First pass: parse XML for all architectures
  for arch, cfg in ARCHS.items():
    print(f"Parsing XML: {cfg['xml']} -> {arch}")
    encodings, enums, types, fmts, op_types_set, suffix_only_ops = parse_xml(cfg["xml"])
    for fmt, ops in FIXES.get(arch, {}).items(): enums.setdefault(fmt, {}).update(ops)
    for fmt, fields in FIELD_FIXES.get(arch, {}).items():
      if fmt in encodings: encodings[fmt] = (encodings[fmt][0] + fields, encodings[fmt][1])
    arch_data[arch] = {"encodings": encodings, "enums": enums, "types": types, "suffix_only_ops": suffix_only_ops}
    for fmt, bits in fmts.items():
      assert fmt not in all_fmts or all_fmts[fmt] == bits, f"FMT_BITS mismatch for {fmt}: {all_fmts[fmt]} vs {bits}"
      all_fmts[fmt] = bits
    all_op_types.update(op_types_set)
  # Write common.py
  common_path = pathlib.Path(__file__).parent / "autogen" / "common.py"
  write_common(all_fmts, all_op_types, common_path)
  print(f"Wrote common.py: {len(all_fmts)} formats, {len(all_op_types)} op types")
  # Write per-arch files from XML
  for arch, data in arch_data.items():
    base = pathlib.Path(__file__).parent / "autogen" / arch
    write_enum(data["enums"], base / "enum.py")
    write_ins(data["encodings"], data["enums"], data["suffix_only_ops"], data["types"], arch, base / "ins.py")
    write_operands(data["types"], data["enums"], arch, base / "operands.py")
    print(f"  {arch}: {len(data['encodings'])} encodings, {sum(len(v) for v in data['enums'].values())} instructions")
  # Second pass: parse PDFs and write pcode
  for arch, cfg in ARCHS.items():
    print(f"Parsing PDF: {arch}...")
    pages = extract_pdf_text(cfg["pdf"])
    name_to_op = {name: op for ops in arch_data[arch]["enums"].values() for op, name in ops.items()}
    pcode = extract_pcode(pages, name_to_op)
    base = pathlib.Path(__file__).parent / "autogen" / arch
    write_pcode(pcode, arch_data[arch]["enums"], arch, base / "str_pcode.py")
    print(f"  {arch}: {len(pcode)} pcode entries")
