import unittest, ctypes
from extra.assembly.amd.autogen.rdna4 import ins as ir4
from extra.assembly.amd.dsl import v, s
from extra.assembly.amd.emu import WaveState, decode_program
from tinygrad.device import Buffer, BufferSpec
from tinygrad.dtype import dtypes

class TestRDNA4Emu(unittest.TestCase):
  def _run(self, insts: list, sgprs: dict[int, int] = None, vgprs: dict[tuple[int, int], int] = None) -> WaveState:
    """Run instructions and return final WaveState."""
    # Add S_ENDPGM if not present
    if not any(isinstance(i, ir4.SOPP) and i.op == ir4.SOPPOp.S_ENDPGM for i in insts):
      insts = list(insts) + [ir4.SOPP(ir4.SOPPOp.S_ENDPGM, simm=0)]

    # Assemble and decode
    code = b''.join(i.to_bytes() for i in insts)
    code_buf = (ctypes.c_uint8 * len(code)).from_buffer_copy(code)
    code_addr = ctypes.addressof(code_buf)
    program_raw = decode_program(code, "rdna4")
    program = {code_addr + offset: val for offset, val in program_raw.items()}

    # Setup wave state
    st = WaveState(n_lanes=1)
    st.pc = code_addr
    if sgprs:
      for idx, val in sgprs.items(): st._write_sgpr(idx, val)
    if vgprs:
      for (reg, lane), val in vgprs.items(): st._write_vgpr(reg, lane, val)

    # Setup vmem buffer with external_ptr=0 (maps to address 0, allows any pointer access)
    vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()

    # Execute
    c_bufs = [ctypes.c_uint64(st.sgpr_buf._buf.va_addr), ctypes.c_uint64(st.vgpr_buf._buf.va_addr),
              ctypes.c_uint64(vmem_buf._buf.va_addr), ctypes.c_uint64(0), ctypes.c_uint64(0)]
    for _ in range(100):
      if (pc := st.pc) == 0xFFFFFFFFFFFFFFFF or pc not in program: break
      _, fxn, globals_list, _ = program[pc]
      fxn(*[c_bufs[g] for g in globals_list])
    return st

  def test_vopd_dual_mov(self):
    """Test VOPD with two V_DUAL_MOV_B32 operations: v[1]=s[1], v[2]=s[2]."""
    insts = [ir4.VOPD(ir4.VOPDOp.V_DUAL_MOV_B32, ir4.VOPDOp.V_DUAL_MOV_B32,
                      vdstx=v[1], vdsty=v[2], srcx0=s[1], srcy0=s[2], vsrcx1=v[0], vsrcy1=v[0])]
    st = self._run(insts, sgprs={1: 0x40e00000, 2: 0x41100000})  # 7.0f, 9.0f
    self.assertEqual(st._read_vgpr(1, 0), 0x40e00000)  # v[1] = 7.0
    self.assertEqual(st._read_vgpr(2, 0), 0x41100000)  # v[2] = 9.0

  def test_vopd_dual_mov_after_other_vopd(self):
    """Test VOPD reuse: first VOPD(v[3]=0, v[0]=?), then VOPD(v[1]=s[1], v[2]=s[2])."""
    # This matches the BEAM kernel sequence that fails
    insts = [
      ir4.VOPD(ir4.VOPDOp.V_DUAL_MOV_B32, ir4.VOPDOp.V_DUAL_MOV_B32,
               vdstx=v[3], vdsty=v[0], srcx0=0, srcy0=s[0], vsrcx1=v[0], vsrcy1=v[0]),  # v[3]=0, v[0]=s[0]
      ir4.VOPD(ir4.VOPDOp.V_DUAL_MOV_B32, ir4.VOPDOp.V_DUAL_MOV_B32,
               vdstx=v[1], vdsty=v[2], srcx0=s[1], srcy0=s[2], vsrcx1=v[0], vsrcy1=v[0]),  # v[1]=s[1], v[2]=s[2]
    ]
    st = self._run(insts, sgprs={0: 0x40a00000, 1: 0x40e00000, 2: 0x41100000})  # 5.0f, 7.0f, 9.0f
    self.assertEqual(st._read_vgpr(1, 0), 0x40e00000)  # v[1] = 7.0
    self.assertEqual(st._read_vgpr(2, 0), 0x41100000)  # v[2] = 9.0

  def test_vopd_with_s_add_f32_sequence(self):
    """Test full BEAM kernel sequence: s_add_f32 then VOPD."""
    # This is the exact sequence from the failing BEAM kernel
    insts = [
      ir4.SOP2(ir4.SOP2Op.S_ADD_F32, sdst=s[0], ssrc0=s[0], ssrc1=s[8]),   # s[0] = s[0] + s[8]
      ir4.SOP2(ir4.SOP2Op.S_ADD_F32, sdst=s[1], ssrc0=s[1], ssrc1=s[9]),   # s[1] = s[1] + s[9]
      ir4.SOP2(ir4.SOP2Op.S_ADD_F32, sdst=s[2], ssrc0=s[2], ssrc1=s[10]),  # s[2] = s[2] + s[10]
      ir4.VOPD(ir4.VOPDOp.V_DUAL_MOV_B32, ir4.VOPDOp.V_DUAL_MOV_B32,
               vdstx=v[3], vdsty=v[0], srcx0=0, srcy0=s[0], vsrcx1=v[0], vsrcy1=v[0]),
      ir4.VOPD(ir4.VOPDOp.V_DUAL_MOV_B32, ir4.VOPDOp.V_DUAL_MOV_B32,
               vdstx=v[1], vdsty=v[2], srcx0=s[1], srcy0=s[2], vsrcx1=v[0], vsrcy1=v[0]),
    ]
    # Input: s[0:2] = [1,2,3], s[8:10] = [4,5,6]
    # After s_add_f32: s[0:2] = [5,7,9]
    st = self._run(insts, sgprs={0: 0x3f800000, 1: 0x40000000, 2: 0x40400000,  # 1.0, 2.0, 3.0
                                  8: 0x40800000, 9: 0x40a00000, 10: 0x40c00000})  # 4.0, 5.0, 6.0
    self.assertEqual(st._read_vgpr(1, 0), 0x40e00000)  # v[1] = 7.0
    self.assertEqual(st._read_vgpr(2, 0), 0x41100000)  # v[2] = 9.0

  def test_s_mov_b32_then_vopd(self):
    """Test s_mov_b32 followed by VOPD - simulates BEAM kernel sequence."""
    # Use s_mov_b32 with SGPR source (copy from pre-initialized SGPRs)
    # s[10:12] will have values set by test harness, copy to s[0:2], then VOPD to VGPRs
    insts = [
      ir4.SOP1(ir4.SOP1Op.S_MOV_B32, sdst=s[0], ssrc0=s[10]),  # s[0] = s[10]
      ir4.SOP1(ir4.SOP1Op.S_MOV_B32, sdst=s[1], ssrc0=s[11]),  # s[1] = s[11]
      ir4.SOP1(ir4.SOP1Op.S_MOV_B32, sdst=s[2], ssrc0=s[12]),  # s[2] = s[12]
      ir4.VOPD(ir4.VOPDOp.V_DUAL_MOV_B32, ir4.VOPDOp.V_DUAL_MOV_B32,
               vdstx=v[1], vdsty=v[2], srcx0=s[1], srcy0=s[2], vsrcx1=v[0], vsrcy1=v[0]),
    ]
    st = self._run(insts, sgprs={10: 0x40a00000, 11: 0x40e00000, 12: 0x41100000})  # 5.0, 7.0, 9.0
    self.assertEqual(st._read_vgpr(1, 0), 0x40e00000)  # v[1] = 7.0
    self.assertEqual(st._read_vgpr(2, 0), 0x41100000)  # v[2] = 9.0

if __name__ == '__main__':
  unittest.main()
