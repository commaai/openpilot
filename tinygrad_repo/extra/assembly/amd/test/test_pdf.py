#!/usr/bin/env python3
"""Test PDF pseudocode extraction from generate.py."""
import unittest
from extra.assembly.amd.generate import extract_pdf_text, extract_pcode, parse_xml, ARCHS, FIXES

EXPECTED_PAGES = {"rdna3": 655, "rdna4": 711, "cdna": 610}

class TestPcodePDF(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.pages = {arch: extract_pdf_text(cfg["pdf"]) for arch, cfg in ARCHS.items()}
    cls.enums = {}
    for arch, cfg in ARCHS.items():
      _, enums, _, _, _, _ = parse_xml(cfg["xml"])
      for fmt, ops in FIXES.get(arch, {}).items(): enums.setdefault(fmt, {}).update(ops)
      cls.enums[arch] = enums
    cls.pcode = {arch: extract_pcode(cls.pages[arch], {n: op for ops in cls.enums[arch].values() for op, n in ops.items()}) for arch in ARCHS}

  def test_page_counts(self):
    for name, exp in EXPECTED_PAGES.items():
      self.assertEqual(len(self.pages[name]), exp, f"{name} page count")

  def test_pcode_extracted(self):
    """Check we extracted a reasonable number of pcode entries."""
    for name in ARCHS:
      self.assertGreater(len(self.pcode[name]), 500, f"{name} pcode count too low")

  def test_pcode_rdna3_tricky(self):
    """Test specific pseudocode patterns that are tricky to extract correctly."""
    pcode = self.pcode['rdna3']
    # BUFFER_ATOMIC_MAX_U64: should have 4 statements (not truncated)
    self.assertEqual(pcode[('BUFFER_ATOMIC_MAX_U64', 72)],
      'tmp = MEM[ADDR].u64;\nsrc = DATA.u64;\nMEM[ADDR].u64 = src >= tmp ? src : tmp;\nRETURN_DATA.u64 = tmp')
    # GLOBAL_STORE_B128: should have 4 MEM stores (not truncated)
    self.assertEqual(pcode[('GLOBAL_STORE_B128', 29)],
      'MEM[ADDR].b32 = VDATA[31 : 0];\nMEM[ADDR + 4U].b32 = VDATA[63 : 32];\nMEM[ADDR + 8U].b32 = VDATA[95 : 64];\nMEM[ADDR + 12U].b32 = VDATA[127 : 96]')
    # S_CMOVK_I32: should have full if/endif block
    self.assertEqual(pcode[('S_CMOVK_I32', 2)],
      "if SCC then\nD0.i32 = 32'I(signext(SIMM16.i16))\nendif")

  def test_pcode_no_examples(self):
    """Pseudocode should not contain example lines with '=>'."""
    for name in ARCHS:
      for (op_name, opcode), code in self.pcode[name].items():
        self.assertNotIn('=>', code, f"{name} {op_name} contains example line with '=>'")

if __name__ == "__main__":
  unittest.main()
