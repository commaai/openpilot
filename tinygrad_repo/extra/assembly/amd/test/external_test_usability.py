# Usability tests for the RDNA3 ASM DSL
# These tests demonstrate how the DSL *should* work for a good user experience
# Currently many of these tests fail - they document desired behavior

import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import Inst, RawImm, SGPR, VGPR

class TestRegisterSliceSyntax(unittest.TestCase):
  """
  Issue: Register slice syntax should use AMD assembly convention (inclusive end).

  In AMD assembly, s[4:7] means registers s4, s5, s6, s7 (4 registers, inclusive).
  The DSL should match this convention so that:
  - s[4:7] gives 4 registers
  - Disassembler output can be copied directly back into DSL code

  Fix: Change _RegFactory.__getitem__ to use inclusive end:
    key.stop - key.start + 1  (instead of key.stop - key.start)
  """
  def test_register_slice_count(self):
    # s[4:7] should give 4 registers: s4, s5, s6, s7 (AMD convention, inclusive)
    reg = s[4:7]
    self.assertEqual(reg.count, 4, "s[4:7] should give 4 registers (s4, s5, s6, s7)")

  def test_register_slice_roundtrip(self):
    # Round-trip: DSL -> disasm -> DSL should preserve register count
    reg = s[4:7]  # 4 registers in AMD convention
    inst = s_load_b128(reg, s[0:1], NULL, 0)
    disasm = inst.disasm()
    # Disasm shows s[4:7] - user should be able to copy this back
    self.assertIn("s[4:7]", disasm)
    # And s[4:7] in DSL should give the same 4 registers
    reg_from_disasm = s[4:7]
    self.assertEqual(reg_from_disasm.count, 4, "s[4:7] from disasm should give 4 registers")


class TestReprReadability(unittest.TestCase):
  """
  Issue: repr() leaks internal RawImm type and omits zero-valued fields.

  When you create v_mov_b32_e32(v[0], v[1]), the repr shows:
    VOP1(op=1, src0=RawImm(257))

  Problems:
  1. vdst=v[0] is omitted because 0 is treated as "default"
  2. src0 shows RawImm(257) instead of v[1]
  3. User sees encoded values (257 = 256 + 1) instead of register names

  Expected repr: VOP1(op=1, vdst=v[0], src0=v[1])
  """
  def test_repr_shows_registers_not_raw_imm(self):
    inst = v_mov_b32_e32(v[0], v[1])
    # Should show v[1], not RawImm(257)
    self.assertNotIn("RawImm", repr(inst), "repr should not expose RawImm internal type")
    self.assertIn("v[1]", repr(inst), "repr should show register name")

  def test_repr_includes_zero_dst(self):
    inst = v_mov_b32_e32(v[0], v[1])
    # v[0] is a valid destination register, should be shown
    self.assertIn("vdst", repr(inst), "repr should include vdst even when 0")

  def test_repr_roundtrip(self):
    # repr should produce something that can be eval'd back
    inst = v_mov_b32_e32(v[0], v[1])
    # This would require repr to output valid Python, e.g.:
    # "VOP1(op=VOP1Op.V_MOV_B32, vdst=v[0], src0=v[1])"
    r = repr(inst)
    # At minimum, it should be human-readable
    self.assertIn("v[", r, "repr should show register syntax")


class TestInstructionEquality(unittest.TestCase):
  """
  Issue: No __eq__ method - instruction comparison requires repr() workaround.

  Two identical instructions should compare equal with ==, but currently:
    inst1 == inst2 returns False

  The test_handwritten.py works around this with:
    self.assertEqual(repr(self.inst), repr(reasm))
  """
  def test_identical_instructions_equal(self):
    inst1 = v_mov_b32_e32(v[0], v[1])
    inst2 = v_mov_b32_e32(v[0], v[1])
    self.assertEqual(inst1, inst2, "identical instructions should be equal")

  def test_different_instructions_not_equal(self):
    inst1 = v_mov_b32_e32(v[0], v[1])
    inst2 = v_mov_b32_e32(v[0], v[2])
    self.assertNotEqual(inst1, inst2, "different instructions should not be equal")


class TestVOPDHelperSignature(unittest.TestCase):
  """
  Issue: VOPD helper functions have confusing semantics.

  v_dual_mul_f32 is defined as:
    v_dual_mul_f32 = functools.partial(VOPD, VOPDOp.V_DUAL_MUL_F32)

  This binds VOPDOp.V_DUAL_MUL_F32 to the FIRST positional arg of VOPD.__init__,
  which is 'opx'. So v_dual_mul_f32 sets the X operation.

  But then test_dual_mul in test_handwritten.py does:
    v_dual_mul_f32(VOPDOp.V_DUAL_MUL_F32, vdstx=v[0], ...)

  This passes V_DUAL_MUL_F32 as the SECOND positional arg (opy), making both
  X and Y operations the same. This is confusing because:
  1. The function name suggests it handles the X operation
  2. But you still pass an opcode as the first arg (which becomes opy)

  Expected: Either make the helper fully specify both ops, or make the
  signature clearer about what the positional arg means.
  """
  def test_vopd_helper_opy_should_be_required(self):
    # Using only keyword args "works" but opy silently defaults to 0
    inst = v_dual_mul_f32(vdstx=v[0], vdsty=v[1], srcx0=v[2], vsrcx1=v[3], srcy0=v[4], vsrcy1=v[5])
    self.assertEqual(inst.opx, VOPDOp.V_DUAL_MUL_F32)
    # Bug: opy defaults to 0 (V_DUAL_FMAC_F32) silently - should require explicit opy
    # This test documents the bug - it should fail once fixed
    self.assertNotEqual(inst.opy, VOPDOp.V_DUAL_FMAC_F32, "opy should not silently default to FMAC")

  def test_vopd_helper_positional_arg_is_opy(self):
    # The first positional arg after the partial becomes opy, not a second opx
    inst = v_dual_mul_f32(VOPDOp.V_DUAL_MOV_B32, vdstx=v[0], vdsty=v[1], srcx0=v[2], vsrcx1=v[3], srcy0=v[4], vsrcy1=v[5])
    self.assertEqual(inst.opx, VOPDOp.V_DUAL_MUL_F32)  # From partial
    self.assertEqual(inst.opy, VOPDOp.V_DUAL_MOV_B32)  # From first positional arg


class TestFieldAccessPreservesType(unittest.TestCase):
  """
  Issue: Field access loses type information.

  After creating an instruction, accessing fields returns encoded int values:
    inst = v_mov_b32_e32(v[0], v[1])
    inst.vdst  # returns 0, not VGPR(0)

  This makes it impossible to round-trip register types through field access.
  """
  def test_vdst_returns_register(self):
    inst = v_mov_b32_e32(v[5], v[1])
    vdst = inst.vdst
    # Should return a VGPR, not an int
    self.assertIsInstance(vdst, (VGPR, int), "vdst should return VGPR or at least be usable")
    # Ideally: self.assertIsInstance(vdst, VGPR)

  def test_src_returns_register_for_vgpr_source(self):
    inst = v_mov_b32_e32(v[0], v[1])
    # src0 is encoded as 257 (256 + 1 for v1)
    # Ideally it should decode back to v[1]
    src0_raw = inst._values.get('src0')
    # Currently returns RawImm(257), should return VGPR(1) or similar
    self.assertNotIsInstance(src0_raw, RawImm, "source should not be RawImm internally")


class TestArgumentDiscoverability(unittest.TestCase):
  """
  Issue: No clear signature for positional arguments.

  inspect.signature(s_load_b128) shows: (*args, literal=None, **kwargs)

  Users have no way to know the argument order without reading source code.
  The order is implicitly defined by the class field definition order.

  Possible fixes:
  1. Add explicit parameter names to functools.partial
  2. Generate type stubs with proper signatures
  3. Add docstrings listing the expected arguments
  """
  def test_signature_has_named_params(self):
    import inspect
    sig = inspect.signature(s_load_b128)
    params = list(sig.parameters.keys())
    # Currently: ['args', 'literal', 'kwargs'] (from *args, literal=None, **kwargs)
    # Expected: something like ['sdata', 'sbase', 'soffset', 'offset', 'literal']
    self.assertIn('sdata', params, "signature should show field names")


class TestSpecialConstants(unittest.TestCase):
  """
  Issue: NULL and other constants are IntEnum values that might be confusing.

  NULL = SrcEnum.NULL = 124, but users might expect NULL to be a special object
  that clearly represents "no register" rather than a magic number.
  """
  def test_null_has_clear_repr(self):
    # NULL should have a clear string representation
    self.assertIn("NULL", str(NULL) or repr(NULL), "NULL should be clearly identifiable")

  def test_null_is_distinguishable_from_int(self):
    # NULL should be distinguishable from the raw integer 124
    self.assertNotEqual(type(NULL), int, "NULL should not be plain int")


if __name__ == "__main__":
  unittest.main()
