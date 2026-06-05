"""Tests for the pcode parser."""
import unittest
from collections import defaultdict
from tinygrad.helpers import DEBUG
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops
from test.mockgpu.amd.emu import parse_pcode
from test.mockgpu.amd.pcode import parse_expr
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE
from tinygrad.runtime.autogen.amd.rdna3.enum import VOP1Op, VOP2Op, SOP2Op, DSOp, GLOBALOp

def _srcs():
  """Create minimal source variables for pcode parsing."""
  def u32(v=0): return UOp.const(dtypes.uint32, v)
  return {'S0': u32(), 'S1': u32(), 'S2': u32(), 'SCC': u32(), 'VCC': UOp.const(dtypes.uint64, 0), 'laneId': u32()}

class TestBasicParsing(unittest.TestCase):
  """Test basic pcode parsing for common instruction patterns."""

  def test_v_add_f32(self):
    """Test parsing V_ADD_F32 pcode."""
    _, assigns = parse_pcode(PCODE[VOP2Op.V_ADD_F32_E32], _srcs())
    self.assertEqual(len(assigns), 1)
    dest, _ = assigns[0]
    self.assertTrue(dest.startswith('D0'))

  def test_v_lshlrev_b32(self):
    """Test parsing V_LSHLREV_B32 pcode."""
    _, assigns = parse_pcode(PCODE[VOP2Op.V_LSHLREV_B32_E32], _srcs())
    self.assertEqual(len(assigns), 1)

  def test_s_cselect_b32(self):
    """Test parsing S_CSELECT_B32 pcode with ternary."""
    _, assigns = parse_pcode(PCODE[SOP2Op.S_CSELECT_B32], _srcs())
    self.assertEqual(len(assigns), 1)

  def test_v_add_co_ci_u32(self):
    """Test parsing V_ADD_CO_CI_U32 with carry."""
    _, assigns = parse_pcode(PCODE[VOP2Op.V_ADD_CO_CI_U32_E32], _srcs())
    self.assertGreaterEqual(len(assigns), 1)

class TestWithSources(unittest.TestCase):
  """Test pcode parsing with actual source operand values."""

  def test_v_add_f32_with_sources(self):
    """Test V_ADD_F32 with actual float constants."""
    s0 = UOp.const(dtypes.uint32, 0x3f800000)  # 1.0f
    s1 = UOp.const(dtypes.uint32, 0x40000000)  # 2.0f
    _, assigns = parse_pcode(PCODE[VOP2Op.V_ADD_F32_E32], {'S0': s0, 'S1': s1})
    self.assertEqual(len(assigns), 1)
    dest, val = assigns[0]
    self.assertTrue(dest.startswith('D0'))
    # Result should be an ADD operation
    self.assertEqual(val.op, Ops.ADD)

  def test_v_mul_f32_with_sources(self):
    """Test V_MUL_F32 with actual float constants."""
    s0 = UOp.const(dtypes.uint32, 0x40000000)  # 2.0f
    s1 = UOp.const(dtypes.uint32, 0x40400000)  # 3.0f
    _, assigns = parse_pcode(PCODE[VOP2Op.V_MUL_F32_E32], {'S0': s0, 'S1': s1})
    self.assertEqual(len(assigns), 1)
    dest, val = assigns[0]
    self.assertEqual(val.op, Ops.MUL)

class TestParseExpr(unittest.TestCase):
  """Test the parse_expr function directly."""

  def test_integer_literals(self):
    """Test parsing integer literals."""
    self.assertEqual(parse_expr('0', {}).arg, 0)
    self.assertEqual(parse_expr('42', {}).arg, 42)
    self.assertEqual(parse_expr('42U', {}).arg, 42)

  def test_negative_integers(self):
    """Test parsing negative integer literals."""
    result = parse_expr('-1', {})
    self.assertEqual(result.arg, -1)
    self.assertEqual(result.dtype, dtypes.int)

  def test_float_literals(self):
    """Test parsing float literals."""
    result = parse_expr('1.0F', {})
    self.assertEqual(result.arg, 1.0)
    self.assertEqual(result.dtype, dtypes.float32)

  def test_hex_literals(self):
    """Test parsing hex literals."""
    result = parse_expr('0xFF', {})
    self.assertEqual(result.arg, 255)

  def test_variable_lookup(self):
    """Test variable lookup in parse_expr."""
    vrs = {'x': UOp.const(dtypes.uint32, 42)}
    result = parse_expr('x', vrs)
    self.assertEqual(result.arg, 42)

  def test_binary_ops(self):
    """Test parsing binary operations."""
    vrs = {'a': UOp.const(dtypes.uint32, 10), 'b': UOp.const(dtypes.uint32, 5)}

    # Addition
    result = parse_expr('a + b', vrs)
    self.assertEqual(result.op, Ops.ADD)

    # Subtraction with constant folding
    result = parse_expr('10 - 5', {})
    self.assertEqual(result.op, Ops.CONST)
    self.assertEqual(result.arg, 5)

  def test_ternary(self):
    """Test parsing ternary expressions."""
    vrs = {'cond': UOp.const(dtypes.bool, True), 'a': UOp.const(dtypes.uint32, 1), 'b': UOp.const(dtypes.uint32, 0)}
    result = parse_expr('cond ? a : b', vrs)
    self.assertEqual(result.op, Ops.WHERE)


class TestForLoopParsing(unittest.TestCase):
  """Test for loop parsing (CLZ/CTZ patterns)."""

  def test_clz_pcode_exists(self):
    """Verify CLZ pcode is available."""
    pcode = PCODE.get(VOP1Op.V_CLZ_I32_U32_E32)
    self.assertIsNotNone(pcode)
    assert pcode is not None
    self.assertIn('for', pcode.lower())

  def test_clz_parsing(self):
    """Test CLZ pcode parsing produces correct structure."""
    pcode = PCODE[VOP1Op.V_CLZ_I32_U32_E32]
    S0 = UOp.const(dtypes.uint32, 0xFFFFFFFF)  # All ones - CLZ should be 0
    _vrs, assigns = parse_pcode(pcode, {'S0': S0})

    self.assertEqual(len(assigns), 1)
    dest, val = assigns[0]
    self.assertTrue(dest.startswith('D0'))
    # Result should be a nested WHERE structure
    self.assertEqual(val.op, Ops.WHERE)

  def test_clz_with_zero(self):
    """Test CLZ with input 0 - should return -1."""
    pcode = PCODE[VOP1Op.V_CLZ_I32_U32_E32]
    S0 = UOp.const(dtypes.uint32, 0)
    _vrs, assigns = parse_pcode(pcode, {'S0': S0})

    # Check that the innermost value (default) is -1 (may be wrapped in CAST)
    val = assigns[0][1]
    # Traverse to innermost WHERE
    while val.op == Ops.WHERE:
      val = val.src[2]  # false branch
    # Unwrap CAST if present
    while val.op == Ops.CAST:
      val = val.src[0]
    self.assertEqual(val.arg, -1)

  def test_ctz_parsing(self):
    """Test CTZ pcode parsing."""
    pcode = PCODE.get(VOP1Op.V_CTZ_I32_B32_E32)
    if pcode is None:
      self.skipTest("V_CTZ_I32_B32_E32 pcode not available")

    S0 = UOp.const(dtypes.uint32, 1)  # LSB set - CTZ should be 0
    _vrs, assigns = parse_pcode(pcode, {'S0': S0})
    self.assertEqual(len(assigns), 1)

class TestDSPcodePatterns(unittest.TestCase):
  """Test DS instruction pcode patterns."""

  def test_global_atomic_add_f32_parsing(self):
    """Test GLOBAL_ATOMIC_ADD_F32 keeps memory values in float dtype."""
    vmem = UOp(Ops.PARAM, dtypes.uint32.ptr(1024), arg=2)
    srcs = {
      'ADDR': UOp.const(dtypes.uint64, 0),
      'DATA': UOp.const(dtypes.uint32, 0x3f800000),
      '_vmem': vmem,
    }

    _, assigns = parse_pcode(PCODE[GLOBALOp.GLOBAL_ATOMIC_ADD_F32], srcs)
    mem_write = next(val for dest, val in assigns if dest == 'MEM[ADDR].f32')
    self.assertEqual(mem_write[1].op, Ops.ADD)  # type: ignore[index]
    self.assertEqual(mem_write[1].dtype, dtypes.float32)  # type: ignore[index]

  def test_ds_load_b32_pcode(self):
    """Test DS_LOAD_B32 pcode is parseable."""
    pcode = PCODE.get(DSOp.DS_LOAD_B32)
    self.assertIsNotNone(pcode)
    assert pcode is not None
    self.assertIn('RETURN_DATA', pcode)
    self.assertIn('MEM[', pcode)

  def test_ds_store_b32_pcode(self):
    """Test DS_STORE_B32 pcode is parseable."""
    pcode = PCODE.get(DSOp.DS_STORE_B32)
    self.assertIsNotNone(pcode)
    assert pcode is not None
    self.assertIn('MEM[', pcode)
    self.assertIn('DATA', pcode)

  def test_mem_read_parsing(self):
    """Test MEM[addr].type read expression parsing."""
    # Create a mock LDS buffer
    lds = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)
    addr = UOp.const(dtypes.uint32, 0)
    vrs = {'_lds': lds, 'ADDR': addr, 'OFFSET': UOp.const(dtypes.uint32, 0)}

    result = parse_expr('MEM[ADDR + OFFSET].b32', vrs)
    # Should be an INDEX operation into LDS
    self.assertIsNotNone(result)

  def test_ds_store_2addr_b32_parsing(self):
    """Test DS_STORE_2ADDR_B32 pcode parsing produces MEM writes."""
    pcode = PCODE.get(DSOp.DS_STORE_2ADDR_B32)
    self.assertIsNotNone(pcode)
    assert pcode is not None
    srcs = {
      'ADDR': UOp.const(dtypes.uint32, 0),
      'OFFSET0': UOp.const(dtypes.uint32, 0),
      'OFFSET1': UOp.const(dtypes.uint32, 1),
      'DATA': UOp.const(dtypes.uint32, 0xAAAAAAAA),
      'DATA2': UOp.const(dtypes.uint32, 0xBBBBBBBB),
    }
    srcs['laneId'] = UOp.const(dtypes.uint32, 0)
    _, assigns = parse_pcode(pcode, srcs)
    # Should have 2 MEM write assignments
    self.assertEqual(len(assigns), 2)
    for dest, val in assigns:
      self.assertTrue(dest.startswith('MEM['))
      # val should be (addr, write_val) tuple
      self.assertIsInstance(val, tuple)
      self.assertEqual(len(val), 2)  # type: ignore[arg-type]

  def test_ds_load_2addr_b32_parsing(self):
    """Test DS_LOAD_2ADDR_B32 pcode parsing produces RETURN_DATA assignments."""
    pcode = PCODE.get(DSOp.DS_LOAD_2ADDR_B32)
    self.assertIsNotNone(pcode)
    assert pcode is not None
    lds = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)
    srcs = {
      'ADDR': UOp.const(dtypes.uint32, 0),
      'OFFSET0': UOp.const(dtypes.uint32, 0),
      'OFFSET1': UOp.const(dtypes.uint32, 1),
      '_lds': lds,
    }
    srcs['laneId'] = UOp.const(dtypes.uint32, 0)
    _, assigns = parse_pcode(pcode, srcs)
    # Should have 2 RETURN_DATA assignments
    self.assertEqual(len(assigns), 2)
    self.assertEqual(assigns[0][0], 'RETURN_DATA[31:0]')
    self.assertEqual(assigns[1][0], 'RETURN_DATA[63:32]')

  def test_ds_store_address_calculation(self):
    """Test DS_STORE_2ADDR_B32 calculates correct addresses (offset * 4)."""
    pcode = PCODE.get(DSOp.DS_STORE_2ADDR_B32)
    assert pcode is not None
    srcs = {
      'ADDR': UOp.const(dtypes.uint32, 100),
      'OFFSET0': UOp.const(dtypes.uint32, 2),
      'OFFSET1': UOp.const(dtypes.uint32, 5),
      'DATA': UOp.const(dtypes.uint32, 0xAAAAAAAA),
      'DATA2': UOp.const(dtypes.uint32, 0xBBBBBBBB),
    }
    srcs['laneId'] = UOp.const(dtypes.uint32, 0)
    _, assigns = parse_pcode(pcode, srcs)
    # Check addresses: 100 + 2*4 = 108, 100 + 5*4 = 120
    # assigns[i][1] is (addr, val) tuple for MEM writes; mypy sees UOp
    self.assertEqual(assigns[0][1][0].simplify().arg, 108)  # type: ignore[index]
    self.assertEqual(assigns[1][1][0].simplify().arg, 120)  # type: ignore[index]

  def test_ds_store_data_values(self):
    """Test DS_STORE_2ADDR_B32 uses correct data values."""
    pcode = PCODE.get(DSOp.DS_STORE_2ADDR_B32)
    assert pcode is not None
    srcs = {
      'ADDR': UOp.const(dtypes.uint32, 0),
      'OFFSET0': UOp.const(dtypes.uint32, 0),
      'OFFSET1': UOp.const(dtypes.uint32, 1),
      'DATA': UOp.const(dtypes.uint32, 0xAAAAAAAA),
      'DATA2': UOp.const(dtypes.uint32, 0xBBBBBBBB),
    }
    srcs['laneId'] = UOp.const(dtypes.uint32, 0)
    _, assigns = parse_pcode(pcode, srcs)
    # assigns[i][1] is (addr, val) tuple for MEM writes; mypy sees UOp
    # DATA[31:0] should preserve the value
    self.assertEqual(assigns[0][1][1].simplify().arg, 0xAAAAAAAA)  # type: ignore[index]
    self.assertEqual(assigns[1][1][1].simplify().arg, 0xBBBBBBBB)  # type: ignore[index]

class TestConditionalParsing(unittest.TestCase):
  """Test conditional (if/elsif/else) pcode parsing."""

  def test_ternary_in_assignment(self):
    """Test parsing ternary expression (which becomes WHERE)."""
    # S_CSELECT_B32: D0.u32 = SCC ? S0.u32 : S1.u32
    pcode = PCODE[SOP2Op.S_CSELECT_B32]
    s0 = UOp.const(dtypes.uint32, 10)
    s1 = UOp.const(dtypes.uint32, 20)
    scc = UOp.const(dtypes.uint32, 1)
    _vrs, assigns = parse_pcode(pcode, {'S0': s0, 'S1': s1, 'SCC': scc})
    self.assertEqual(len(assigns), 1)
    dest, val = assigns[0]
    self.assertTrue(dest.startswith('D0'))
    # Result should be a WHERE (ternary becomes WHERE)
    self.assertEqual(val.op, Ops.WHERE)

class TestConcatWidthParsing(unittest.TestCase):
  """Test that bit extracts keep the right width for concat/unary ops."""

  def test_permlanex16_altrow_concat(self):
    for row, expected in [(0, 1), (1, 0), (2, 3), (3, 2)]:
      parsed = parse_expr('{ row[1], ~row[0] }', {'row': UOp.const(dtypes.uint32, row)})
      self.assertEqual(parsed.simplify().arg, expected)

  def test_permlane64_altlane_concat(self):
    for lane, expected in [(0, 32), (1, 33), (31, 63), (32, 0), (63, 31)]:
      parsed = parse_expr('{ ~lane[5], lane[4:0] }', {'lane': UOp.const(dtypes.uint32, lane)})
      self.assertEqual(parsed.simplify().arg, expected)

  def test_permlane64_wave64_pcode_indices(self):
    vgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(256), arg=0)
    srcs = {
      'SRC0': UOp.const(dtypes.uint32, 0),
      'VDST': UOp.const(dtypes.uint32, 1),
      'EXEC_LO': UOp.const(dtypes.uint32, 0xFFFFFFFF),
      'EXEC': UOp.const(dtypes.uint64, 0xFFFFFFFFFFFFFFFF),
      '_vgpr': vgpr,
      '_wave_size': 64,
      'S0': UOp.const(dtypes.uint32, 0),
      'S1': UOp.const(dtypes.uint32, 0),
      'S2': UOp.const(dtypes.uint32, 0),
    }

    def load_idx(v: UOp) -> int:
      simp = v.simplify()
      self.assertEqual(simp.op, Ops.LOAD)
      self.assertEqual(simp.src[0].op, Ops.INDEX)
      idx = simp.src[0].src[1].simplify()
      self.assertEqual(idx.op, Ops.CONST)
      return idx.arg

    _, assigns = parse_pcode(PCODE[VOP1Op.V_PERMLANE64_B32_E32], srcs)
    self.assertEqual(len(assigns), 64)
    for lane, (dst_idx, src_idx) in {0: (64, 32), 31: (95, 63), 32: (96, 0), 63: (127, 31)}.items():
      self.assertEqual(assigns[lane][1][0].simplify().arg, dst_idx)  # type: ignore[index]
      self.assertEqual(load_idx(assigns[lane][1][1]), src_idx)  # type: ignore[index]

class TestAllPcode(unittest.TestCase):
  """Test that all pcode from all architectures can be parsed."""

  def _make_srcs(self):
    """Create dummy source variables for pcode parsing."""
    u32, u64 = lambda v=0: UOp.const(dtypes.uint32, v), lambda v=0: UOp.const(dtypes.uint64, v)
    lds = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)
    return {'laneId': u32(), 'laneID': u32(), 'S0': u32(), 'S1': u32(), 'S2': u32(), 'S3': u32(), 'SRC0': u32(),
            'D0': u32(), 'D1': u32(), 'DST': u32(), 'VDST': u32(), 'SDST': u32(),
            'VCC': u64(), 'VCCZ': u32(), 'EXEC': u64(), 'EXEC_LO': u32(), 'EXECZ': u32(), 'SCC': u32(),
            'SIMM16': u32(), 'SIMM32': u32(), 'OFFSET': u32(), 'OFFSET0': u32(), 'OFFSET1': u32(), 'offset1': u32(),
            'ADDR': u32(), 'ADDR_BASE': u32(), 'TADDR': u32(), 'DATA': u32(), 'DATA0': u32(), 'DATA1': u32(), 'DATA2': u32(),
            'VDATA': u32(), 'VDATA0': u32(), 'VDATA1': u32(), 'VDATA2': u32(), 'VDATA3': u32(),
            'OPSEL': u32(), 'OPSEL_HI': u32(), 'NEG': u32(), 'NEG_HI': u32(), 'CLAMP': u32(),
            'M0': u32(), 'PC': u64(), 'DENORM': u32(1), 'ROUND_MODE': u32(), 'ROUND_TOWARD_ZERO': u32(),
            'ROUND_NEAREST_EVEN': u32(), 'WAVE_STATUS': u32(),
            'MAX_FLOAT_F32': u32(0x7f7fffff), 'Unsigned': u32(1), 'clampedLOD': u32(),
            '_lds': lds, '_vmem': lds, '_active': UOp.const(dtypes.bool, True)}

  def _parse_all_pcode(self, pcode_dict, arch: str, min_pct: float):
    """Parse all pcode. RuntimeError = parser limitation (ok), other exceptions = real bugs."""
    srcs = self._make_srcs()
    passed, skipped, errors = 0, 0, defaultdict(list)
    for op, pcode in pcode_dict.items():
      try:
        parse_pcode(pcode, srcs)
        passed += 1
      except RuntimeError as e:
        skipped += 1
        errors[str(e)].append(op.name)
      except Exception as e: self.fail(f"[{arch}] {op.name}: {e}\nPcode: {pcode[:200]}")
    total = len(pcode_dict)
    pct = 100 * passed / total
    print(f"{arch}: {passed}/{total} ({pct:.1f}%) parsed, {skipped} skipped")
    if DEBUG >= 2:
      for err, ops in sorted(errors.items(), key=lambda x: -len(x[1])):
        print(f"  {err}: {', '.join(ops[:5])}{'...' if len(ops) > 5 else ''} ({len(ops)})")
    self.assertGreaterEqual(pct, min_pct, f"[{arch}] {pct:.1f}% < {min_pct}% threshold")

  def test_parse_all_cdna_pcode(self):
    from tinygrad.runtime.autogen.amd.cdna.str_pcode import PCODE as CDNA_PCODE
    self._parse_all_pcode(CDNA_PCODE, "CDNA", min_pct=60)

  def test_parse_all_rdna3_pcode(self):
    from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE as RDNA3_PCODE
    self._parse_all_pcode(RDNA3_PCODE, "RDNA3", min_pct=90)

  def test_parse_all_rdna4_pcode(self):
    from tinygrad.runtime.autogen.amd.rdna4.str_pcode import PCODE as RDNA4_PCODE
    self._parse_all_pcode(RDNA4_PCODE, "RDNA4", min_pct=65)

if __name__ == "__main__":
  unittest.main()
