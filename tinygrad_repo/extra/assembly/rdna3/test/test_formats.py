#!/usr/bin/env python3
"""Test MUBUF, MTBUF, MIMG, EXP, DS formats against LLVM."""
import unittest
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.lib import encode_src

class TestMUBUF(unittest.TestCase):
  """Test MUBUF (buffer) instructions."""

  def test_buffer_load_b32_basic(self):
    # buffer_load_b32 v5, off, s[8:11], s3 offset:4095
    # GFX11: encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x02,0x03]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0x50,0xe0,0x00,0x05,0x02,0x03]))

  def test_buffer_load_b32_idxen(self):
    # buffer_load_b32 v5, v0, s[8:11], s3 idxen offset:4095
    # GFX11: encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0x03]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, idxen=1)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0x03]))

  def test_buffer_load_b32_offen(self):
    # buffer_load_b32 v5, v0, s[8:11], s3 offen offset:4095
    # GFX11: encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x42,0x03]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, offen=1)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0x50,0xe0,0x00,0x05,0x42,0x03]))

  def test_buffer_load_b32_glc(self):
    # buffer_load_b32 v5, off, s[8:11], s3 offset:4095 glc
    # GFX11: encoding: [0xff,0x4f,0x50,0xe0,0x00,0x05,0x02,0x03]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, glc=1)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x4f,0x50,0xe0,0x00,0x05,0x02,0x03]))

  def test_buffer_load_b32_slc(self):
    # buffer_load_b32 v5, off, s[8:11], s3 offset:4095 slc
    # GFX11: encoding: [0xff,0x1f,0x50,0xe0,0x00,0x05,0x02,0x03]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, slc=1)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x1f,0x50,0xe0,0x00,0x05,0x02,0x03]))

  def test_buffer_load_b32_dlc(self):
    # buffer_load_b32 v5, off, s[8:11], s3 offset:4095 dlc
    # GFX11: encoding: [0xff,0x2f,0x50,0xe0,0x00,0x05,0x02,0x03]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, dlc=1)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x2f,0x50,0xe0,0x00,0x05,0x02,0x03]))

  def test_buffer_load_b32_all_flags(self):
    # buffer_load_b32 v5, off, s[8:11], s3 offset:4095 glc slc dlc
    # GFX11: encoding: [0xff,0x7f,0x50,0xe0,0x00,0x05,0x02,0x03]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, glc=1, slc=1, dlc=1)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x7f,0x50,0xe0,0x00,0x05,0x02,0x03]))

  def test_buffer_store_b32(self):
    # buffer_store_b32 v1, off, s[12:15], s4 offset:4095
    # GFX11: encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x03,0x04]
    inst = buffer_store_b32(vdata=v[1], vaddr=v[0], srsrc=s[12:16], soffset=s[4], offset=4095)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0x68,0xe0,0x00,0x01,0x03,0x04]))

  def test_buffer_load_b64(self):
    # buffer_load_b64 v[5:6], off, s[8:11], s3 offset:4095
    # GFX11: encoding: [0xff,0x0f,0x54,0xe0,0x00,0x05,0x02,0x03]
    inst = buffer_load_b64(vdata=v[5:7], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0x54,0xe0,0x00,0x05,0x02,0x03]))

  def test_buffer_load_soffset_m0(self):
    # buffer_load_b32 v5, off, s[8:11], m0 offset:4095
    # GFX11: encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x02,0x7d]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=M0, offset=4095)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0x50,0xe0,0x00,0x05,0x02,0x7d]))

  def test_buffer_load_soffset_inline_const(self):
    # buffer_load_b32 v5, off, s[8:11], 0 offset:4095
    # GFX11: encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x02,0x80]
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=0, offset=4095)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0x50,0xe0,0x00,0x05,0x02,0x80]))

  def test_buffer_disasm_roundtrip(self):
    inst = buffer_load_b32(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, glc=1)
    decoded = MUBUF.from_bytes(inst.to_bytes())
    self.assertEqual(decoded.to_bytes(), inst.to_bytes())


class TestMTBUF(unittest.TestCase):
  """Test MTBUF (typed buffer) instructions."""

  def test_tbuffer_load_format_x(self):
    # tbuffer_load_format_x v5, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT] offset:4095
    # BUF_FMT_32_FLOAT = 22
    # GFX11: encoding: [0xff,0x0f,0xb0,0xe8,0x00,0x05,0x02,0x03]
    inst = tbuffer_load_format_x(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, format=22)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0xb0,0xe8,0x00,0x05,0x02,0x03]))

  def test_tbuffer_store_format_x(self):
    # tbuffer_store_format_x v5, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT] offset:4095
    # BUF_FMT_32_FLOAT = 22
    # GFX11: encoding: [0xff,0x0f,0xb2,0xe8,0x00,0x05,0x02,0x03]
    inst = tbuffer_store_format_x(vdata=v[5], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, format=22)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x0f,0xb2,0xe8,0x00,0x05,0x02,0x03]))

  def test_tbuffer_load_format_xy(self):
    # tbuffer_load_format_xy v[5:6], off, s[8:11], s3 format:[BUF_FMT_32_32_FLOAT] offset:4095
    # BUF_FMT_32_32_FLOAT = 50
    # GFX11: encoding: [0xff,0x8f,0x90,0xe9,0x00,0x05,0x02,0x03]
    inst = tbuffer_load_format_xy(vdata=v[5:7], vaddr=v[0], srsrc=s[8:12], soffset=s[3], offset=4095, format=50)
    self.assertEqual(inst.to_bytes(), bytes([0xff,0x8f,0x90,0xe9,0x00,0x05,0x02,0x03]))


class TestMIMG(unittest.TestCase):
  """Test MIMG (image) instructions."""

  def test_image_load_2d(self):
    # image_load v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D
    # GFX11: encoding: [0x04,0x0f,0x00,0xf0,0x04,0x00,0x00,0x00]
    inst = image_load(vdata=v[0:4], vaddr=v[4:6], srsrc=s[0:8], dmask=0xf, dim=1)  # dim=1 is SQ_RSRC_IMG_2D
    self.assertEqual(inst.to_bytes(), bytes([0x04,0x0f,0x00,0xf0,0x04,0x00,0x00,0x00]))

  def test_image_store_2d(self):
    # image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D
    # GFX11: encoding: [0x04,0x0f,0x18,0xf0,0x04,0x00,0x00,0x00]
    inst = image_store(vdata=v[0:4], vaddr=v[4:6], srsrc=s[0:8], dmask=0xf, dim=1)
    self.assertEqual(inst.to_bytes(), bytes([0x04,0x0f,0x18,0xf0,0x04,0x00,0x00,0x00]))

  def test_image_load_1d(self):
    # image_load v[0:3], v4, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
    # GFX11: encoding: [0x00,0x0f,0x00,0xf0,0x04,0x00,0x00,0x00]
    inst = image_load(vdata=v[0:4], vaddr=v[4], srsrc=s[0:8], dmask=0xf, dim=0)  # dim=0 is SQ_RSRC_IMG_1D
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x0f,0x00,0xf0,0x04,0x00,0x00,0x00]))

  def test_image_sample(self):
    # image_sample v[0:3], v[4:5], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    # GFX11: encoding: [0x04,0x0f,0x6c,0xf0,0x04,0x00,0x00,0x08]
    inst = image_sample(vdata=v[0:4], vaddr=v[4:6], srsrc=s[0:8], ssamp=s[8:12], dmask=0xf, dim=1)
    self.assertEqual(inst.to_bytes(), bytes([0x04,0x0f,0x6c,0xf0,0x04,0x00,0x00,0x08]))

  def test_image_load_d16(self):
    # image_load v[0:1], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D d16
    # GFX11: encoding: [0x04,0x0f,0x02,0xf0,0x04,0x00,0x00,0x00]
    inst = image_load(vdata=v[0:2], vaddr=v[4:6], srsrc=s[0:8], dmask=0xf, dim=1, d16=1)
    self.assertEqual(inst.to_bytes(), bytes([0x04,0x0f,0x02,0xf0,0x04,0x00,0x00,0x00]))


class TestEXP(unittest.TestCase):
  """Test EXP (export) instructions."""

  def test_exp_mrt0(self):
    # exp mrt0 v0, v1, v2, v3
    # GFX11: encoding: [0x0f,0x00,0x00,0xf8,0x00,0x01,0x02,0x03]
    inst = EXP(en=0xf, target=0, vsrc0=v[0], vsrc1=v[1], vsrc2=v[2], vsrc3=v[3])
    self.assertEqual(inst.to_bytes(), bytes([0x0f,0x00,0x00,0xf8,0x00,0x01,0x02,0x03]))

  def test_exp_mrtz(self):
    # exp mrtz v4, v3, v2, v1
    # GFX11: encoding: [0x8f,0x00,0x00,0xf8,0x04,0x03,0x02,0x01]
    inst = EXP(en=0xf, target=8, vsrc0=v[4], vsrc1=v[3], vsrc2=v[2], vsrc3=v[1])
    self.assertEqual(inst.to_bytes(), bytes([0x8f,0x00,0x00,0xf8,0x04,0x03,0x02,0x01]))

  def test_exp_mrtz_done(self):
    # exp mrtz v4, v3, v2, v1 done
    # GFX11: encoding: [0x8f,0x08,0x00,0xf8,0x04,0x03,0x02,0x01]
    inst = EXP(en=0xf, target=8, vsrc0=v[4], vsrc1=v[3], vsrc2=v[2], vsrc3=v[3], done=1)
    self.assertEqual(inst.to_bytes(), bytes([0x8f,0x08,0x00,0xf8,0x04,0x03,0x02,0x03]))

  def test_exp_partial_mask(self):
    # exp mrt0 v0, v1, off, off (en=0x3, only first two components)
    # GFX11: encoding: [0x03,0x00,0x00,0xf8,0x00,0x01,0x00,0x00]
    inst = EXP(en=0x3, target=0, vsrc0=v[0], vsrc1=v[1], vsrc2=v[0], vsrc3=v[0])
    self.assertEqual(inst.to_bytes(), bytes([0x03,0x00,0x00,0xf8,0x00,0x01,0x00,0x00]))

  def test_exp_row_en(self):
    # exp mrtz v4, v3, v2, v1 row_en
    # GFX11: encoding: [0x8f,0x20,0x00,0xf8,0x04,0x03,0x02,0x01]
    inst = EXP(en=0xf, target=8, vsrc0=v[4], vsrc1=v[3], vsrc2=v[2], vsrc3=v[1], row=1)
    self.assertEqual(inst.to_bytes(), bytes([0x8f,0x20,0x00,0xf8,0x04,0x03,0x02,0x01]))


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
    inst = ds_load_b64(vdst=v[0:2], addr=v[2])
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
    inst = global_load_b32(vdst=v[0], addr=v[1:3], saddr=OFF)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x52,0xdc,0x01,0x00,0x7c,0x00]))

  def test_global_store_b32(self):
    # global_store_b32 v[0:1], v2, off (seg=2 for global)
    # GFX11: encoding: [0x00,0x00,0x6a,0xdc,0x00,0x02,0x7c,0x00]
    inst = global_store_b32(addr=v[0:2], data=v[2], saddr=OFF)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x6a,0xdc,0x00,0x02,0x7c,0x00]))

  def test_global_load_b32_saddr(self):
    # global_load_b32 v0, v1, s[0:1] (seg=2 for global)
    # GFX11: encoding: [0x00,0x00,0x52,0xdc,0x01,0x00,0x00,0x00]
    inst = global_load_b32(vdst=v[0], addr=v[1], saddr=s[0:2])
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x52,0xdc,0x01,0x00,0x00,0x00]))

  def test_global_load_b32_offset(self):
    # global_load_b32 v0, v[1:2], off offset:256 (seg=2 for global)
    # GFX11: encoding: [0x00,0x01,0x52,0xdc,0x01,0x00,0x7c,0x00]
    inst = global_load_b32(vdst=v[0], addr=v[1:3], saddr=OFF, offset=256)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x01,0x52,0xdc,0x01,0x00,0x7c,0x00]))

  def test_global_load_b64(self):
    # global_load_b64 v[0:1], v[2:3], off (seg=2 for global)
    # GFX11: encoding: [0x00,0x00,0x56,0xdc,0x02,0x00,0x7c,0x00]
    inst = global_load_b64(vdst=v[0:2], addr=v[2:4], saddr=OFF)
    self.assertEqual(inst.to_bytes(), bytes([0x00,0x00,0x56,0xdc,0x02,0x00,0x7c,0x00]))


class TestSMEM(unittest.TestCase):
  """Test SMEM (scalar memory) instructions - regression tests for glc/dlc bit positions."""

  def test_smem_dlc_bit_position(self):
    # s_load_b32 s5, s[2:3], s0 dlc - tests that DLC is at bit 13 (not bit 14)
    # GFX11: encoding: [0x41,0x21,0x00,0xf4,0x00,0x00,0x00,0x00]
    inst = s_load_b32(sdata=s[5], sbase=s[2], soffset=s[0], dlc=1)
    self.assertEqual(inst.to_bytes(), bytes([0x41,0x21,0x00,0xf4,0x00,0x00,0x00,0x00]))

  def test_smem_glc_bit_position(self):
    # s_load_b32 s5, s[2:3], s0 glc - tests that GLC is at bit 14 (not bit 16)
    # GFX11: encoding: [0x41,0x41,0x00,0xf4,0x00,0x00,0x00,0x00]
    inst = s_load_b32(sdata=s[5], sbase=s[2], soffset=s[0], glc=1)
    self.assertEqual(inst.to_bytes(), bytes([0x41,0x41,0x00,0xf4,0x00,0x00,0x00,0x00]))

  def test_smem_glc_dlc_combined(self):
    # s_load_b32 s5, s[2:3], s0 glc dlc - tests both flags together
    # GFX11: encoding: [0x41,0x61,0x00,0xf4,0x00,0x00,0x00,0x00]
    inst = s_load_b32(sdata=s[5], sbase=s[2], soffset=s[0], glc=1, dlc=1)
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
    from extra.assembly.rdna3.lib import RawImm
    inst = VOP3(VOP3Op.V_ADD3_U32, vdst=v[5], src0=RawImm(107), src1=0xaf123456, src2=v[255])
    expected = bytes([0x05,0x00,0x55,0xd6,0x6b,0xfe,0xfd,0x07,0x56,0x34,0x12,0xaf])
    self.assertEqual(inst.to_bytes(), expected)

  def test_vop3_literal_null_operand(self):
    # v_add3_u32 v5, null, exec_lo, 0xaf123456
    # GFX11: encoding: [0x05,0x00,0x55,0xd6,0x7c,0xfc,0xfc,0x03,0x56,0x34,0x12,0xaf]
    from extra.assembly.rdna3.lib import RawImm
    inst = VOP3(VOP3Op.V_ADD3_U32, vdst=v[5], src0=NULL, src1=RawImm(126), src2=0xaf123456)
    expected = bytes([0x05,0x00,0x55,0xd6,0x7c,0xfc,0xfc,0x03,0x56,0x34,0x12,0xaf])
    self.assertEqual(inst.to_bytes(), expected)

  def test_vop3p_with_literal(self):
    # Test VOP3P literal encoding (also uses Inst64)
    from extra.assembly.rdna3.lib import RawImm
    inst = VOP3P(VOP3POp.V_PK_ADD_F16, vdst=v[5], src0=RawImm(240), src1=0x12345678, src2=v[0])
    self.assertEqual(len(inst.to_bytes()), 12)  # 8 bytes + 4 byte literal


if __name__ == "__main__":
  unittest.main()
