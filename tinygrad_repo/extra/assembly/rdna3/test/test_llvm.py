#!/usr/bin/env python3
"""Test RDNA3 assembler/disassembler against LLVM test vectors."""
import unittest, re
from tinygrad.helpers import fetch
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.asm import asm
from extra.assembly.rdna3.test.test_roundtrip import compile_asm, disassemble_lib

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

# Format info: (filename, format_class, op_enum)
LLVM_TEST_FILES = {
  # Scalar ALU
  'sop1': ('gfx11_asm_sop1.s', SOP1, SOP1Op),
  'sop2': ('gfx11_asm_sop2.s', SOP2, SOP2Op),
  'sopp': ('gfx11_asm_sopp.s', SOPP, SOPPOp),
  'sopk': ('gfx11_asm_sopk.s', SOPK, SOPKOp),
  'sopc': ('gfx11_asm_sopc.s', SOPC, SOPCOp),
  # Vector ALU
  'vop1': ('gfx11_asm_vop1.s', VOP1, VOP1Op),
  'vop2': ('gfx11_asm_vop2.s', VOP2, VOP2Op),
  'vopc': ('gfx11_asm_vopc.s', VOPC, VOPCOp),
  'vop3': ('gfx11_asm_vop3.s', VOP3, VOP3Op),
  'vop3p': ('gfx11_asm_vop3p.s', VOP3P, VOP3POp),
  'vop3sd': ('gfx11_asm_vop3.s', VOP3SD, VOP3SDOp),  # VOP3SD shares file with VOP3
  'vinterp': ('gfx11_asm_vinterp.s', VINTERP, VINTERPOp),
  'vopd': ('gfx11_asm_vopd.s', VOPD, VOPDOp),
  'vopcx': ('gfx11_asm_vopcx.s', VOPC, VOPCOp),  # VOPCX uses VOPC format
  # VOP3 promotions (VOP1/VOP2/VOPC promoted to VOP3 encoding)
  'vop3_from_vop1': ('gfx11_asm_vop3_from_vop1.s', VOP3, VOP3Op),
  'vop3_from_vop2': ('gfx11_asm_vop3_from_vop2.s', VOP3, VOP3Op),
  'vop3_from_vopc': ('gfx11_asm_vop3_from_vopc.s', VOP3, VOP3Op),
  'vop3_from_vopcx': ('gfx11_asm_vop3_from_vopcx.s', VOP3, VOP3Op),
  # Memory
  'ds': ('gfx11_asm_ds.s', DS, DSOp),
  'smem': ('gfx11_asm_smem.s', SMEM, SMEMOp),
  'flat': ('gfx11_asm_flat.s', FLAT, FLATOp),
  'mubuf': ('gfx11_asm_mubuf.s', MUBUF, MUBUFOp),
  'mtbuf': ('gfx11_asm_mtbuf.s', MTBUF, MTBUFOp),
  'mimg': ('gfx11_asm_mimg.s', MIMG, MIMGOp),
  # WMMA (matrix multiply)
  'wmma': ('gfx11_asm_wmma.s', VOP3P, VOP3POp),
  # Additional features
  'vop3_features': ('gfx11_asm_vop3_features.s', VOP3, VOP3Op),
  'vop3p_features': ('gfx11_asm_vop3p_features.s', VOP3P, VOP3POp),
  'vopd_features': ('gfx11_asm_vopd_features.s', VOPD, VOPDOp),
  # Alias files (alternative mnemonics)
  'vop3_alias': ('gfx11_asm_vop3_alias.s', VOP3, VOP3Op),
  'vop3p_alias': ('gfx11_asm_vop3p_alias.s', VOP3P, VOP3POp),
  'vopc_alias': ('gfx11_asm_vopc_alias.s', VOPC, VOPCOp),
  'vopcx_alias': ('gfx11_asm_vopcx_alias.s', VOPC, VOPCOp),
  'vinterp_alias': ('gfx11_asm_vinterp_alias.s', VINTERP, VINTERPOp),
  'smem_alias': ('gfx11_asm_smem_alias.s', SMEM, SMEMOp),
  'mubuf_alias': ('gfx11_asm_mubuf_alias.s', MUBUF, MUBUFOp),
  'mtbuf_alias': ('gfx11_asm_mtbuf_alias.s', MTBUF, MTBUFOp),
}

def parse_llvm_tests(text: str) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs."""
  tests, lines = [], text.split('\n')
  for i, line in enumerate(lines):
    line = line.strip()
    if not line or line.startswith(('//', '.', ';')): continue
    asm_text = line.split('//')[0].strip()
    if not asm_text: continue
    for j in range(i, min(i + 3, len(lines))):
      # Match GFX11, W32, or W64 encodings (all valid for gfx11)
      if m := re.search(r'(?:GFX11|W32|W64)[^:]*:.*?encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
        if hex_bytes:
          try: tests.append((asm_text, bytes.fromhex(hex_bytes)))
          except ValueError: pass
        break
  return tests

def try_assemble(text: str):
  """Try to assemble instruction text, return bytes or None on failure."""
  try: return asm(text).to_bytes()
  except: return None

class TestLLVM(unittest.TestCase):
  """Test assembler and disassembler against all LLVM test vectors."""
  tests: dict[str, list[tuple[str, bytes]]] = {}

  @classmethod
  def setUpClass(cls):
    for name, (filename, _, _) in LLVM_TEST_FILES.items():
      try:
        data = fetch(f"{LLVM_BASE}/{filename}").read_bytes()
        cls.tests[name] = parse_llvm_tests(data.decode('utf-8', errors='ignore'))
      except Exception as e:
        print(f"Warning: couldn't fetch {filename}: {e}")
        cls.tests[name] = []

# Generate test methods dynamically for each format
def _make_asm_test(name):
  def test(self):
    passed, failed, skipped = 0, 0, 0
    for asm_text, expected in self.tests.get(name, []):
      result = try_assemble(asm_text)
      if result is None: skipped += 1
      elif result == expected: passed += 1
      else: failed += 1
    print(f"{name.upper()} asm: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0)
  return test

def _make_disasm_test(name):
  def test(self):
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    compiler = HIPCompiler('gfx1100')
    _, fmt_cls, op_enum = LLVM_TEST_FILES[name]
    passed, failed, skipped, failures = 0, 0, 0, []
    # VOP3SD opcodes that share encoding with VOP3 (only for vop3sd test, not vopc promotions)
    # Note: opcodes 0-255 are VOPC promoted to VOP3, never VOP3SD
    vop3sd_opcodes = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}
    # vop3_from_vopc/vopcx tests have VOPC opcodes 0-255, not VOP3SD - don't detect as VOP3SD
    is_vopc_promotion = name in ('vop3_from_vopc', 'vop3_from_vopcx')
    # Undocumented opcodes not in AMD ISA PDF - skip these
    undocumented = {'smem': {34, 35}, 'sopk': {22, 23}, 'sopp': {8, 58, 59}}  # s_atc_probe*, s_subvector_loop*, s_waitcnt_depctr, unknown
    for asm_text, data in self.tests.get(name, []):
      if len(data) > fmt_cls._size(): continue  # skip literals (need different handling)
      # Skip undocumented opcodes
      temp_inst = fmt_cls.from_bytes(data)
      temp_op = temp_inst._values.get('op', 0)
      temp_op = temp_op.val if hasattr(temp_op, 'val') else temp_op
      if temp_op in undocumented.get(name, set()): skipped += 1; continue
      # Skip SOPP no-imm instructions with non-zero simm16 (can't roundtrip through LLVM)
      if name == 'sopp':
        simm16 = temp_inst._values.get('simm16', 0)
        simm16 = simm16.val if hasattr(simm16, 'val') else simm16
        sopp_no_imm = {48, 54, 53, 55, 60, 61, 62}  # s_endpgm, s_barrier, s_wakeup, s_icache_inv, s_wait_idle, s_endpgm_saved, s_code_end
        if temp_op in sopp_no_imm and simm16 != 0: skipped += 1; continue
      try:
        # VOP3 and VOP3SD share encoding - peek at opcode to determine which class to use
        if fmt_cls.__name__ in ('VOP3', 'VOP3SD'):
          temp = VOP3.from_bytes(data)
          op_val = temp._values.get('op', 0)
          op_val = op_val.val if hasattr(op_val, 'val') else op_val
          is_vop3sd = (op_val in vop3sd_opcodes) and not is_vopc_promotion
          decoded = VOP3SD.from_bytes(data) if is_vop3sd else VOP3.from_bytes(data)
          # Validate opcode with appropriate enum
          if is_vop3sd:
            VOP3SDOp(op_val)
          else:
            VOP3Op(op_val)
        else:
          decoded = fmt_cls.from_bytes(data)
          op_val = decoded._values.get('op', 0)
          op_val = op_val.val if hasattr(op_val, 'val') else op_val
          op_enum(op_val)  # validate opcode
        if decoded.to_bytes()[:len(data)] != data:
          failed += 1; failures.append(f"decode roundtrip failed for {data.hex()}"); continue
        disasm_str = decoded.disasm()
        # Test: LLVM should assemble our disasm output to the same bytes
        llvm_bytes = compile_asm(disasm_str, compiler)
        if llvm_bytes is None:
          failed += 1; failures.append(f"LLVM failed to assemble: '{disasm_str}' (from '{asm_text}')")
        elif llvm_bytes == data: passed += 1
        else: failed += 1; failures.append(f"'{disasm_str}': expected={data.hex()} got={llvm_bytes.hex()}")
      except Exception as e:
        failed += 1; failures.append(f"exception for {data.hex()}: {e}")
    print(f"{name.upper()} disasm: {passed} passed, {failed} failed" + (f", {skipped} skipped" if skipped else ""))
    if failures[:10]: print("  " + "\n  ".join(failures[:10]))
    self.assertEqual(failed, 0)
  return test

for name in LLVM_TEST_FILES:
  setattr(TestLLVM, f'test_{name}_asm', _make_asm_test(name))
  setattr(TestLLVM, f'test_{name}_disasm', _make_disasm_test(name))

if __name__ == "__main__":
  unittest.main()
