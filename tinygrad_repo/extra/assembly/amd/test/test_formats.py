#!/usr/bin/env python3
"""Test DS and other compute-relevant instruction formats.

Note: Graphics-only formats (EXP, MUBUF, MTBUF, MIMG) are not supported - use GLOBAL/FLAT for memory access in compute.
"""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import VCC_HI, EXEC_LO, NULL
OFF = NULL  # OFF is alias for NULL
from extra.assembly.amd import detect_format


class TestDS(unittest.TestCase):
  """Test DS (data share / LDS) instructions."""

  def test_ds_store_b32(self):
    # ds_store_b32 v0, v1
    # GFX11: encoding: [0x00,0x00,0x34,0xd8,0x00,0x01,0x00,0x00]
    inst = ds_store_b32(addr=v[0], data0=v[1])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x34,0xd8,0x00,0x01,0x00,0x00]))

  def test_ds_load_b32(self):
    # ds_load_b32 v0, v1
    # GFX11: encoding: [0x00,0x00,0xd8,0xd8,0x01,0x00,0x00,0x00]
    inst = ds_load_b32(vdst=v[0], addr=v[1])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0xd8,0xd8,0x01,0x00,0x00,0x00]))

  def test_ds_store_b32_offset(self):
    # ds_store_b32 v0, v1 offset:64
    # GFX11: encoding: [0x40,0x00,0x34,0xd8,0x00,0x01,0x00,0x00]
    inst = ds_store_b32(addr=v[0], data0=v[1], offset0=64)
    self.assertEqual(inst.to_bytes(), bytes([0x40,0x00,0x34,0xd8,0x00,0x01,0x00,0x00]))

  def test_ds_load_b64(self):
    # ds_load_b64 v[0:1], v2
    # GFX11: encoding: [0x00,0x00,0xd8,0xd9,0x02,0x00,0x00,0x00]
    inst = ds_load_b64(vdst=v[0:1], addr=v[2])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0xd8,0xd9,0x02,0x00,0x00,0x00]))

  def test_ds_add_u32(self):
    # ds_add_u32 v0, v1
    # GFX11: encoding: [0x00,0x00,0x00,0xd8,0x00,0x01,0x00,0x00]
    inst = ds_add_u32(addr=v[0], data0=v[1])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x00,0xd8,0x00,0x01,0x00,0x00]))

  def test_ds_store_b32_gds(self):
    # ds_store_b32 v0, v1 gds
    # GFX11: encoding: [0x00,0x00,0x36,0xd8,0x00,0x01,0x00,0x00]
    inst = ds_store_b32(addr=v[0], data0=v[1], gds=1)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x36,0xd8,0x00,0x01,0x00,0x00]))


class TestVOP3(unittest.TestCase):
  """Test VOP3 (3-operand vector) instructions."""

  def test_v_fma_f32(self):
    # v_fma_f32 v0, v1, v2, v3
    # GFX11: encoding: [0x00,0x00,0x13,0xd6,0x01,0x05,0x0e,0x04]
    inst = v_fma_f32(vdst=v[0], src0=v[1], src1=v[2], src2=v[3])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x13,0xd6,0x01,0x05,0x0e,0x04]))

  def test_v_mad_f32(self):
    # v_fmac_f32_e64 v0, v1, v2 (fmac is fma with implicit dst as src2)
    # Use v_fma_f32 with vdst == src2
    inst = v_fma_f32(vdst=v[0], src0=v[1], src1=v[2], src2=v[0])
    self.assertEqual(inst.to_bytes()[:4], bytes([0x00,0x00,0x13,0xd6]))

  def test_v_add3_u32(self):
    # v_add3_u32 v0, v1, v2, v3
    # GFX11: encoding: [0x00,0x00,0x55,0xd6,0x01,0x05,0x0e,0x04]
    inst = v_add3_u32(vdst=v[0], src0=v[1], src1=v[2], src2=v[3])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x55,0xd6,0x01,0x05,0x0e,0x04]))


class TestFLAT(unittest.TestCase):
  """Test FLAT/GLOBAL/SCRATCH memory instructions."""

  def test_global_load_b32(self):
    # global_load_b32 v0, v[1:2], off (seg=2 for global)
    # GFX11: encoding: [0x00,0x00,0x52,0xdc,0x01,0x00,0x7c,0x00]
    inst = global_load_b32(vdst=v[0], addr=v[1:2], saddr=OFF)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x52,0xdc,0x01,0x00,0x7c,0x00]))

  def test_global_store_b32(self):
    # global_store_b32 v[0:1], v2, off (seg=2 for global)
    # GFX11: encoding: [0x00,0x00,0x6a,0xdc,0x00,0x02,0x7c,0x00]
    inst = global_store_b32(addr=v[0:1], data=v[2], saddr=OFF)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x6a,0xdc,0x00,0x02,0x7c,0x00]))

  def test_global_load_b32_saddr(self):
    # global_load_b32 v0, v1, s[0:1] (seg=2 for global)
    # GFX11: encoding: [0x00,0x00,0x52,0xdc,0x01,0x00,0x00,0x00]
    inst = global_load_b32(vdst=v[0], addr=v[1], saddr=s[0:1])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x52,0xdc,0x01,0x00,0x00,0x00]))

  def test_global_load_b32_offset(self):
    # global_load_b32 v0, v[1:2], off offset:256 (seg=2 for global)
    # GFX11: encoding: [0x00,0x01,0x52,0xdc,0x01,0x00,0x7c,0x00]
    inst = global_load_b32(vdst=v[0], addr=v[1:2], saddr=OFF, offset=256)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x01,0x52,0xdc,0x01,0x00,0x7c,0x00]))

  def test_global_load_b64(self):
    # global_load_b64 v[0:1], v[2:3], off (seg=2 for global)
    # GFX11: encoding: [0x00,0x00,0x56,0xdc,0x02,0x00,0x7c,0x00]
    inst = global_load_b64(vdst=v[0:1], addr=v[2:3], saddr=OFF)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x56,0xdc,0x02,0x00,0x7c,0x00]))


class TestSMEM(unittest.TestCase):
  """Test SMEM (scalar memory) instructions - regression tests for glc/dlc bit positions."""

  def test_smem_dlc_bit_position(self):
    # s_load_b32 s5, s[2:3], s0 dlc - tests that DLC is at bit 13 (not bit 14)
    # GFX11: encoding: [0x41,0x21,0x00,0xf4,0x00,0x00,0x00,0x00]
    inst = s_load_b32(sdata=s[5], sbase=s[2:3], soffset=s[0], dlc=1)
    self.assertEqual(inst.to_bytes(), bytes([0x41,0x21,0x00,0xf4,0x00,0x00,0x00,0x00]))

  def test_smem_glc_bit_position(self):
    # s_load_b32 s5, s[2:3], s0 glc - tests that GLC is at bit 14 (not bit 16)
    # GFX11: encoding: [0x41,0x41,0x00,0xf4,0x00,0x00,0x00,0x00]
    inst = s_load_b32(sdata=s[5], sbase=s[2:3], soffset=s[0], glc=1)
    self.assertEqual(inst.to_bytes(), bytes([0x41,0x41,0x00,0xf4,0x00,0x00,0x00,0x00]))

  def test_smem_glc_dlc_combined(self):
    # s_load_b32 s5, s[2:3], s0 glc dlc - tests both flags together
    # GFX11: encoding: [0x41,0x61,0x00,0xf4,0x00,0x00,0x00,0x00]
    inst = s_load_b32(sdata=s[5], sbase=s[2:3], soffset=s[0], glc=1, dlc=1)
    self.assertEqual(inst.to_bytes(), bytes([0x41,0x61,0x00,0xf4,0x00,0x00,0x00,0x00]))

  def test_smem_disasm_roundtrip_dlc(self):
    # Test that disassembly/reassembly preserves DLC bit correctly
    data = bytes([0x41,0x21,0x00,0xf4,0x00,0x00,0x00,0x00])
    decoded = SMEM.from_bytes(data)
    self.assertEqual(decoded.to_bytes(), data)

  def test_smem_disasm_roundtrip_glc_dlc(self):
    # Test that disassembly/reassembly preserves GLC+DLC bits correctly
    data = bytes([0x41,0x61,0x00,0xf4,0x00,0x00,0x00,0x00])
    decoded = SMEM.from_bytes(data)
    self.assertEqual(decoded.to_bytes(), data)


class TestVOP3Literal(unittest.TestCase):
  """Test VOP3 literal handling - regression tests for Inst64 literal encoding."""

  def test_vop3_with_literal(self):
    # v_add3_u32 v5, vcc_hi, 0xaf123456, v255
    # GFX11: encoding: [0x05,0x00,0x55,0xd6,0x6b,0xfe,0xfd,0x07,0x56,0x34,0x12,0xaf]
    inst = VOP3(VOP3Op.V_ADD3_U32, vdst=v[5], src0=VCC_HI, src1=0xaf123456, src2=v[255])
    expected = bytes([0x05,0x00,0x55,0xd6,0x6b,0xfe,0xfd,0x07,0x56,0x34,0x12,0xaf])
    self.assertEqual(inst.to_bytes(), expected)

  def test_vop3_literal_null_operand(self):
    # v_add3_u32 v5, null, exec_lo, 0xaf123456
    # GFX11: encoding: [0x05,0x00,0x55,0xd6,0x7c,0xfc,0xfc,0x03,0x56,0x34,0x12,0xaf]
    inst = VOP3(VOP3Op.V_ADD3_U32, vdst=v[5], src0=NULL, src1=EXEC_LO, src2=0xaf123456)
    expected = bytes([0x05,0x00,0x55,0xd6,0x7c,0xfc,0xfc,0x03,0x56,0x34,0x12,0xaf])
    self.assertEqual(inst.to_bytes(), expected)

  def test_vop3p_with_literal(self):
    # Test VOP3P literal encoding (also uses Inst64)
    inst = VOP3P(VOP3POp.V_PK_ADD_F16, vdst=v[5], src0=0.5, src1=0x12345678, src2=v[0])
    self.assertEqual(len(inst.to_bytes()), 12)  # 8 bytes + 4 byte literal


class TestDetectFormat(unittest.TestCase):
  """Test detect_format uses encoding from autogen classes."""

  def test_detect_sopp(self):
    self.assertEqual(detect_format(s_endpgm().to_bytes()), SOPP)
    self.assertEqual(detect_format(s_nop(0).to_bytes()), SOPP)
    self.assertEqual(detect_format(s_barrier().to_bytes()), SOPP)

  def test_detect_sop1(self):
    self.assertEqual(detect_format(s_mov_b32(s[0], 0).to_bytes()), SOP1)
    self.assertEqual(detect_format(s_mov_b64(s[0:1], 0).to_bytes()), SOP1)

  def test_detect_sop2(self):
    self.assertEqual(detect_format(s_add_u32(s[0], s[1], s[2]).to_bytes()), SOP2)
    self.assertEqual(detect_format(s_mul_i32(s[0], s[1], s[2]).to_bytes()), SOP2)

  def test_detect_sopc(self):
    self.assertEqual(detect_format(s_cmp_eq_i32(s[0], s[1]).to_bytes()), SOPC)

  def test_detect_sopk(self):
    self.assertEqual(detect_format(s_movk_i32(s[0], 0x1234).to_bytes()), SOPK)

  def test_detect_vop1(self):
    self.assertEqual(detect_format(v_mov_b32_e32(v[0], 0).to_bytes()), VOP1)
    self.assertEqual(detect_format(v_rcp_f32_e32(v[0], v[1]).to_bytes()), VOP1)

  def test_detect_vop2(self):
    self.assertEqual(detect_format(v_add_f32_e32(v[0], v[1], v[2]).to_bytes()), VOP2)
    self.assertEqual(detect_format(v_mul_f32_e32(v[0], v[1], v[2]).to_bytes()), VOP2)

  def test_detect_vopc(self):
    self.assertEqual(detect_format(v_cmp_eq_f32_e32(v[0], v[1]).to_bytes()), VOPC)
    self.assertEqual(detect_format(v_cmp_lt_i32_e32(v[0], v[1]).to_bytes()), VOPC)

  def test_detect_vop3(self):
    self.assertEqual(detect_format(v_add_f32_e64(v[0], v[1], v[2]).to_bytes()), VOP3)
    self.assertEqual(detect_format(v_fma_f32(v[0], v[1], v[2], v[3]).to_bytes()), VOP3)

  def test_detect_vop3p(self):
    self.assertEqual(detect_format(VOP3P(VOP3POp.V_PK_ADD_F16, v[0], v[1], v[2], v[3]).to_bytes()), VOP3P)

  def test_detect_smem(self):
    self.assertEqual(detect_format(s_load_b32(sdata=s[0], sbase=s[2:3], offset=0).to_bytes()), SMEM)
    self.assertEqual(detect_format(s_load_b64(sdata=s[0:1], sbase=s[2:3], soffset=s[5]).to_bytes()), SMEM)

  def test_detect_ds(self):
    self.assertEqual(detect_format(ds_load_b32(v[0], v[1]).to_bytes()), DS)
    self.assertEqual(detect_format(ds_store_b32(v[0], v[1]).to_bytes()), DS)

  def test_detect_flat(self):
    self.assertEqual(detect_format(global_load_b32(vdst=v[0], addr=v[1:2], saddr=NULL).to_bytes()), GLOBAL)
    self.assertEqual(detect_format(global_store_b32(addr=v[0:1], data=v[2], saddr=NULL).to_bytes()), GLOBAL)

  def test_detect_vopd(self):
    inst = VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[0], vdsty=v[1], srcx0=0, srcy0=0)
    self.assertEqual(detect_format(inst.to_bytes()), VOPD)

  def test_detect_vinterp(self):
    inst = VINTERP(VINTERPOp.V_INTERP_P10_F32, vdst=v[0], src0=v[1], src1=v[2], src2=v[3])
    self.assertEqual(detect_format(inst.to_bytes()), VINTERP)


if __name__ == "__main__":
  unittest.main()
