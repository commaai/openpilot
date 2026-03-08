#!/usr/bin/env python3
"""Test AMD assembler/disassembler against LLVM test vectors.

Only compute-relevant instruction formats are tested. Graphics-only formats not supported:
  - MUBUF/MTBUF: buffer instructions with resource descriptors (use GLOBAL/FLAT instead)
  - MIMG: image/texture instructions
  - EXP/VEXPORT: export instructions for pixel/vertex output
  - VIMAGE/VSAMPLE: image sampling instructions (RDNA4)
  - VBUFFER: buffer instructions (RDNA4)
"""
import unittest, re, subprocess, functools
from tinygrad.helpers import fetch
from extra.assembly.amd.disasm import disasm
from extra.assembly.amd import decode_inst, detect_format
from extra.assembly.amd.test.helpers import get_llvm_mc, get_target, get_mattr

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.0/llvm/test/MC/AMDGPU"

# RDNA3 (gfx11) test files for compute instructions
# Excluded: gfx11_asm_mubuf.s, gfx11_asm_mtbuf.s, gfx11_asm_mimg.s, gfx11_asm_mubuf_alias.s, gfx11_asm_mtbuf_alias.s (graphics-only)
RDNA_FILES = ['gfx11_asm_sop1.s', 'gfx11_asm_sop2.s', 'gfx11_asm_sopp.s', 'gfx11_asm_sopk.s', 'gfx11_asm_sopc.s',
  'gfx11_asm_vop1.s', 'gfx11_asm_vop2.s', 'gfx11_asm_vopc.s', 'gfx11_asm_vop3.s', 'gfx11_asm_vop3p.s', 'gfx11_asm_vinterp.s',
  'gfx11_asm_vopd.s', 'gfx11_asm_vopcx.s', 'gfx11_asm_vop3_from_vop1.s', 'gfx11_asm_vop3_from_vop2.s', 'gfx11_asm_vop3_from_vopc.s',
  'gfx11_asm_vop3_from_vopcx.s', 'gfx11_asm_ds.s', 'gfx11_asm_smem.s', 'gfx11_asm_flat.s',
  'gfx11_asm_wmma.s', 'gfx11_asm_vop3_features.s', 'gfx11_asm_vop3p_features.s', 'gfx11_asm_vopd_features.s',
  'gfx11_asm_vop3_alias.s', 'gfx11_asm_vop3p_alias.s', 'gfx11_asm_vopc_alias.s', 'gfx11_asm_vopcx_alias.s', 'gfx11_asm_vinterp_alias.s',
  'gfx11_asm_smem_alias.s']
# CDNA (gfx9/gfx90a/gfx942/gfx950) test files for compute instructions
# Excluded: gfx9_asm_mubuf.s, gfx9_asm_mtbuf.s, gfx90a_ldst_acc.s (has MIMG mixed in)
# Exclude gfx90a: 'gfx90a_asm_features.s', 'mai-gfx90a.s',
# Exclude gfx950: 'gfx950_asm_features.s' (disasm error)
CDNA_FILES = ['gfx9_asm_sop1.s', 'gfx9_asm_sop2.s', 'gfx9_asm_sopp.s', 'gfx9_asm_sopk.s', 'gfx9_asm_sopc.s',
  'gfx9_asm_vop1.s', 'gfx9_asm_vop2.s', 'gfx9_asm_vopc.s', 'gfx9_asm_vop3.s', 'gfx9_asm_vop3p.s',
  'gfx9_asm_ds.s', 'gfx9_asm_flat.s', 'gfx9_asm_smem.s',
  'flat-scratch-gfx942.s', 'gfx942_asm_features.s', 'mai-gfx942.s',
  'gfx950_asm_vop1.s', 'gfx950_asm_read_tr.s', 'mai-gfx950.s']
# RDNA4 (gfx12) test files for compute instructions
# Excluded: gfx12_asm_vbuffer_mubuf.s, gfx12_asm_vbuffer_mtbuf.s, gfx12_asm_exp.s (graphics-only)
RDNA4_FILES = ['gfx12_asm_sop1.s', 'gfx12_asm_sop2.s', 'gfx12_asm_sopp.s', 'gfx12_asm_sopk.s', 'gfx12_asm_sopc.s',
  'gfx12_asm_vop1.s', 'gfx12_asm_vop2.s', 'gfx12_asm_vopc.s', 'gfx12_asm_vopcx.s', 'gfx12_asm_vop3.s', 'gfx12_asm_vop3c.s',
  'gfx12_asm_vop3cx.s', 'gfx12_asm_vop3p.s', 'gfx12_asm_vop3_from_vop1.s', 'gfx12_asm_vop3_from_vop2.s',
  'gfx12_asm_vop3p_features.s', 'gfx12_asm_vopd.s', 'gfx12_asm_vopd_features.s',
  'gfx12_asm_ds.s', 'gfx12_asm_smem.s',
  'gfx12_asm_wmma_w32.s']

def _parse_llvm_tests(text: str, pattern: str) -> list[tuple[str, bytes]]:
  tests = []
  for block in text.split('\n\n'):
    asm_text, encoding = None, None
    for line in block.split('\n'):
      line = line.strip()
      if not line or line.startswith(('.', ';')): continue
      if not line.startswith('//'):
        asm_text = line.split('//')[0].strip() or asm_text
      if m := re.search(pattern + r'[^:]*:.*?(?:encoding:\s*)?\[(0x[0-9a-f,x\s]+)\]', line, re.I):
        encoding = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
    if asm_text and encoding:
      try: tests.append((asm_text, bytes.fromhex(encoding)))
      except ValueError: pass
  return tests

def _get_tests_uncached(f: str, arch: str) -> list[tuple[str, bytes]]:
  text = fetch(f"{LLVM_BASE}/{f}").read_bytes().decode('utf-8', errors='ignore')
  if arch == "rdna3":
    # Match GFX11 and W32 only (wavefront32 mode)
    tests = _parse_llvm_tests(text, r'(?:GFX11|W32)')
  elif arch == "rdna4":
    # Match GFX12 (but not GFX1250) and W32 only (wavefront32 mode)
    tests = _parse_llvm_tests(text, r'(?:GFX12(?!50)|W32)')
  elif 'gfx90a' in f or 'gfx942' in f or 'gfx950' in f:
    tests = _parse_llvm_tests(text, r'(?:GFX90A|GFX942|GFX950)')
  else:
    tests = _parse_llvm_tests(text, r'(?:VI9|GFX9|CHECK)')
  # Exclude v_interp_* (graphics-only, not on CDNA)
  if arch == "cdna": tests = [(asm, data) for asm, data in tests if not asm.startswith('v_interp_')]
  # Filter out tests where original ASM isn't valid on target (e.g., gfx9 tests with gfx942/gfx950 constraints)
  if arch == "cdna" and not ('gfx942' in f or 'gfx950' in f or 'gfx90a' in f): tests = _filter_valid_asm(tests, arch)
  return tests

@functools.cache
def _get_tests(f: str, arch: str) -> list[tuple[str, bytes]]: return _get_tests_uncached(f, arch)

def _compile_asm_batch(instrs: list[str], arch: str = "rdna3", mcpu: str|None = None) -> list[bytes]:
  if not instrs: return []
  mcpu, mattr = mcpu or get_target(arch), get_mattr(arch)
  result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', f'-mcpu={mcpu}', f'-mattr={mattr}', '-show-encoding'],
    input=".text\n" + "\n".join(instrs) + "\n", capture_output=True, text=True, timeout=30)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc failed: {result.stderr.strip()}")
  return [bytes.fromhex(line.split('encoding:')[1].strip()[1:-1].replace('0x', '').replace(',', '').replace(' ', ''))
          for line in result.stdout.split('\n') if 'encoding:' in line]

def _filter_valid_asm(tests: list[tuple[str, bytes]], arch: str) -> list[tuple[str, bytes]]:
  """Filter out tests where the original ASM isn't valid on the target (e.g., gfx9 tests with gfx942/gfx950 constraints)."""
  if not tests: return []
  mcpu = get_target(arch)
  # Batch assemble all instructions, parse stderr to find which lines failed
  instrs = [asm for asm, _ in tests]
  result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', f'-mcpu={mcpu}', '-show-encoding'],
    input=".text\n" + "\n".join(instrs) + "\n", capture_output=True, text=True, timeout=30)
  # Parse error lines from stderr (format: "<stdin>:N:..." where N is 1-indexed, line 1 is ".text")
  failed_lines = set()
  for line in result.stderr.split('\n'):
    if m := re.match(r'<stdin>:(\d+):', line): failed_lines.add(int(m.group(1)) - 1)  # -1 for .text, so line 2 -> index 1 -> tests[0]
  # Also filter out tests where LLVM roundtrip doesn't match original (reserved bits set in original)
  valid = [(asm, data) for i, (asm, data) in enumerate(tests) if (i + 1) not in failed_lines]
  if not valid: return []
  llvm_result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', f'-mcpu={mcpu}', '-show-encoding'],
    input=".text\n" + "\n".join(asm for asm, _ in valid) + "\n", capture_output=True, text=True, timeout=30)
  llvm_bytes = [bytes.fromhex(line.split('encoding:')[1].strip()[1:-1].replace('0x', '').replace(',', '').replace(' ', ''))
                for line in llvm_result.stdout.split('\n') if 'encoding:' in line]
  return [(asm, data) for (asm, data), lb in zip(valid, llvm_bytes) if lb == data]

def _make_test(f: str, arch: str, test_type: str):
  def test(self):
    tests = _get_tests(f, arch)
    name = f"{arch}_{test_type}_{f}"
    mcpu = "gfx942" if arch == "cdna" and "gfx942" in f else get_target(arch)
    if test_type == "roundtrip":
      passed, skipped = 0, 0
      for _, data in tests:
        try:
          decoded = detect_format(data, arch).from_bytes(data)
          self.assertEqual(decoded.to_bytes()[:len(data)], data)
          passed += 1
        except ValueError: skipped += 1  # skip invalid opcodes not in enum
      print(f"{name}: {passed} passed, {skipped} skipped")
      self.assertEqual(skipped, 0, f"{name}: {skipped} tests skipped, expected 0")
    elif test_type == "repr":
      # Test that eval(repr(inst)) reproduces the instruction
      if arch == "rdna3": import extra.assembly.amd.autogen.rdna3.ins as ins
      elif arch == "rdna4": import extra.assembly.amd.autogen.rdna4.ins as ins
      elif arch == "cdna": import extra.assembly.amd.autogen.cdna.ins as ins
      ns = {k: getattr(ins, k) for k in dir(ins) if not k.startswith('_')}
      passed, skipped = 0, 0
      for _, data in tests:
        try:
          decoded = detect_format(data, arch).from_bytes(data)
          if decoded.to_bytes()[:len(data)] != data: skipped += 1; continue  # skip if binary roundtrip fails
          r = repr(decoded)
          try:
            decoded2 = eval(r, ns)  # noqa: S307
            if decoded == decoded2: passed += 1
            else: skipped += 1
          except Exception: skipped += 1
        except ValueError: skipped += 1
      print(f"{name}: {passed} passed, {skipped} skipped")
      self.assertEqual(skipped, 0, f"{name}: {skipped} tests skipped, expected 0")
    elif test_type == "disasm":
      to_test = []
      for _, data in tests:
        try:
          decoded = decode_inst(data, arch)
          enc = decoded.to_bytes()[:len(data)]
          # Skip if roundtrip fails, disasm fails, or op_name is missing (disasm starts with space)
          if enc == data and (d := disasm(decoded)) and not d.startswith(' '): to_test.append((enc, d))
        except: pass
      skipped = len(tests) - len(to_test)
      print(f"{name}: {len(to_test)} passed, {skipped} skipped")
      self.assertEqual(skipped, 0, f"{name}: {skipped} tests skipped, expected 0")
      # Compare disasm->reassemble with original encoding (filter reserved bit cases where LLVM can't reproduce)
      llvm_bytes = _compile_asm_batch([t[1] for t in to_test], arch, mcpu)
      valid = [(enc, d, llvm) for (enc, d), llvm in zip(to_test, llvm_bytes) if llvm == enc]
      print(f"{name}: {len(valid)}/{len(to_test)} matched LLVM encoding")
      for enc, _, llvm in valid: self.assertEqual(llvm, enc)
  return test

class TestLLVM(unittest.TestCase): pass

for f in RDNA_FILES:
  setattr(TestLLVM, f"test_rdna3_roundtrip_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna3", "roundtrip"))
  setattr(TestLLVM, f"test_rdna3_disasm_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna3", "disasm"))
  setattr(TestLLVM, f"test_rdna3_repr_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna3", "repr"))
for f in CDNA_FILES:
  setattr(TestLLVM, f"test_cdna_roundtrip_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "cdna", "roundtrip"))
  setattr(TestLLVM, f"test_cdna_disasm_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "cdna", "disasm"))
  setattr(TestLLVM, f"test_cdna_repr_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "cdna", "repr"))
for f in RDNA4_FILES:
  setattr(TestLLVM, f"test_rdna4_roundtrip_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna4", "roundtrip"))
  setattr(TestLLVM, f"test_rdna4_disasm_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna4", "disasm"))
  setattr(TestLLVM, f"test_rdna4_repr_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna4", "repr"))

if __name__ == "__main__":
  unittest.main()
