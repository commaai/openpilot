#!/usr/bin/env python3
"""Test that PDF parser correctly extracts format fields."""
import unittest
from extra.assembly.rdna3.autogen import (
  SOP1, SOP2, SOPK, SOPP, VOP1, VOP2, VOP3SD, VOPC, FLAT, VOPD,
  SOP1Op, SOP2Op, VOP1Op, VOP3Op
)

# expected formats with key fields and whether they have ENCODING
EXPECTED_FORMATS = {
  'DPP16': (['SRC0', 'DPP_CTRL', 'BANK_MASK', 'ROW_MASK'], False),
  'DPP8': (['SRC0', 'LANE_SEL0', 'LANE_SEL7'], False),
  'DS': (['OP', 'ADDR', 'DATA0', 'DATA1', 'VDST'], True),
  'EXP': (['EN', 'TARGET', 'VSRC0', 'VSRC1', 'VSRC2', 'VSRC3'], True),
  'FLAT': (['OP', 'ADDR', 'DATA', 'SADDR', 'VDST', 'OFFSET'], True),
  'LDSDIR': (['VDST', 'OP'], True),
  'MIMG': (['OP', 'VADDR', 'VDATA', 'SRSRC', 'DMASK'], True),
  'MTBUF': (['OP', 'VADDR', 'VDATA', 'SRSRC', 'FORMAT', 'SOFFSET'], True),
  'MUBUF': (['OP', 'VADDR', 'VDATA', 'SRSRC', 'SOFFSET'], True),
  'SMEM': (['OP', 'SBASE', 'SDATA', 'OFFSET', 'SOFFSET'], True),
  'SOP1': (['OP', 'SDST', 'SSRC0'], True),
  'SOP2': (['OP', 'SDST', 'SSRC0', 'SSRC1'], True),
  'SOPC': (['OP', 'SSRC0', 'SSRC1'], True),
  'SOPK': (['OP', 'SDST', 'SIMM16'], True),
  'SOPP': (['OP', 'SIMM16'], True),
  'VINTERP': (['OP', 'VDST', 'SRC0', 'SRC1', 'SRC2'], True),
  'VOP1': (['OP', 'VDST', 'SRC0'], True),
  'VOP2': (['OP', 'VDST', 'SRC0', 'VSRC1'], True),
  'VOP3': (['OP', 'VDST', 'SRC0', 'SRC1', 'SRC2'], True),
  'VOP3P': (['OP', 'VDST', 'SRC0', 'SRC1', 'SRC2'], True),
  'VOP3SD': (['OP', 'VDST', 'SDST', 'SRC0', 'SRC1', 'SRC2'], True),
  'VOPC': (['OP', 'SRC0', 'VSRC1'], True),
  'VOPD': (['OPX', 'OPY', 'SRCX0', 'SRCY0', 'VDSTX', 'VDSTY'], True),
}

class TestPDFParserGenerate(unittest.TestCase):
  """Test the PDF parser by running generate() and checking results."""
  result: dict

  @classmethod
  def setUpClass(cls):
    from extra.assembly.rdna3.gen import generate
    cls.result = generate()

  def test_all_formats_present(self):
    """All expected formats should be parsed."""
    for fmt_name in EXPECTED_FORMATS:
      self.assertIn(fmt_name, self.result["formats"], f"missing format {fmt_name}")

  def test_format_count(self):
    """Should have exactly 23 formats."""
    self.assertEqual(len(self.result["formats"]), 23)

  def test_no_duplicate_fields(self):
    """No format should have duplicate field names."""
    for fmt_name, fields in self.result["formats"].items():
      field_names = [f[0] for f in fields]
      self.assertEqual(len(field_names), len(set(field_names)), f"{fmt_name} has duplicate fields: {field_names}")

  def test_expected_fields(self):
    """Each format should have its expected key fields."""
    for fmt_name, (expected_fields, has_encoding) in EXPECTED_FORMATS.items():
      fields = {f[0] for f in self.result["formats"].get(fmt_name, [])}
      for field in expected_fields:
        self.assertIn(field, fields, f"{fmt_name} missing {field}")
      if has_encoding:
        self.assertIn("ENCODING", fields, f"{fmt_name} should have ENCODING")
      else:
        self.assertNotIn("ENCODING", fields, f"{fmt_name} should not have ENCODING")

  def test_vopd_no_dpp16_fields(self):
    """VOPD should not have DPP16-specific fields (parser boundary bug)."""
    vopd_fields = {f[0] for f in self.result["formats"].get("VOPD", [])}
    for field in ['DPP_CTRL', 'BANK_MASK', 'ROW_MASK']:
      self.assertNotIn(field, vopd_fields, f"VOPD should not have {field}")

  def test_dpp16_no_vinterp_fields(self):
    """DPP16 should not have VINTERP-specific fields."""
    dpp16_fields = {f[0] for f in self.result["formats"].get("DPP16", [])}
    for field in ['VDST', 'WAITEXP']:
      self.assertNotIn(field, dpp16_fields, f"DPP16 should not have {field}")

  def test_sopp_no_smem_fields(self):
    """SOPP should not have SMEM fields (page break bug)."""
    sopp_fields = {f[0] for f in self.result["formats"].get("SOPP", [])}
    for field in ['SBASE', 'SDATA']:
      self.assertNotIn(field, sopp_fields, f"SOPP should not have {field}")

class TestPDFParser(unittest.TestCase):
  """Verify format classes have correct fields from PDF parsing."""

  def test_sop2_fields(self):
    """SOP2 should have op, sdst, ssrc0, ssrc1."""
    for field in ['op', 'sdst', 'ssrc0', 'ssrc1']:
      self.assertIn(field, SOP2._fields)
    self.assertEqual(SOP2._fields['op'].hi, 29)
    self.assertEqual(SOP2._fields['op'].lo, 23)

  def test_sop1_fields(self):
    """SOP1 should have op, sdst, ssrc0 with correct bit positions."""
    for field in ['op', 'sdst', 'ssrc0']:
      self.assertIn(field, SOP1._fields)
    self.assertNotIn('simm16', SOP1._fields)
    self.assertEqual(SOP1._fields['ssrc0'].hi, 7)
    self.assertEqual(SOP1._fields['ssrc0'].lo, 0)
    assert SOP1._encoding is not None
    self.assertEqual(SOP1._encoding[0].hi, 31)
    self.assertEqual(SOP1._encoding[1], 0b101111101)

  def test_vop3sd_fields(self):
    """VOP3SD should have all fields including src0/src1/src2 from page continuation."""
    for field in ['op', 'vdst', 'sdst', 'src0', 'src1', 'src2']:
      self.assertIn(field, VOP3SD._fields)
    self.assertEqual(VOP3SD._fields['src0'].hi, 40)
    self.assertEqual(VOP3SD._fields['src0'].lo, 32)
    self.assertEqual(VOP3SD._size(), 8)

  def test_flat_has_vdst(self):
    """FLAT should have vdst field."""
    self.assertIn('vdst', FLAT._fields)
    self.assertEqual(FLAT._fields['vdst'].hi, 63)
    self.assertEqual(FLAT._fields['vdst'].lo, 56)

  def test_encoding_bits(self):
    """Verify encoding bits are correct for major formats."""
    tests = [
      (SOP2, 31, 30, 0b10),
      (SOPK, 31, 28, 0b1011),
      (SOPP, 31, 23, 0b101111111),
      (VOP1, 31, 25, 0b0111111),
      (VOP2, 31, 31, 0b0),
      (VOPC, 31, 25, 0b0111110),
      (FLAT, 31, 26, 0b110111),
    ]
    for cls, hi, lo, val in tests:
      assert cls._encoding is not None
      self.assertEqual(cls._encoding[0].hi, hi, f"{cls.__name__} encoding hi")
      self.assertEqual(cls._encoding[0].lo, lo, f"{cls.__name__} encoding lo")
      self.assertEqual(cls._encoding[1], val, f"{cls.__name__} encoding val")

  def test_opcode_enums_exist(self):
    """Verify opcode enums are generated with expected counts."""
    self.assertGreater(len(SOP1Op), 50)
    self.assertGreater(len(SOP2Op), 50)
    self.assertGreater(len(VOP1Op), 50)
    self.assertGreater(len(VOP3Op), 200)

  def test_vopd_no_duplicate_fields(self):
    """VOPD should not have duplicate fields and should not include DPP16 fields."""
    field_names = list(VOPD._fields.keys())
    self.assertEqual(len(field_names), len(set(field_names)))
    for field in ['srcx0', 'srcy0', 'opx', 'opy']:
      self.assertIn(field, VOPD._fields)
    for field in ['dpp_ctrl', 'bank_mask', 'row_mask']:
      self.assertNotIn(field, VOPD._fields)

if __name__ == "__main__":
  unittest.main()
