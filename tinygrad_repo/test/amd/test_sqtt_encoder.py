#!/usr/bin/env python3
"""Tests for SQTT encoder: verifies the emulator produces correct SQTT traces for known kernels.

Run with: DEV=MOCK+AMD python -m pytest test/amd/test_sqtt_encoder.py -v
"""
import ctypes, unittest
from tinygrad.helpers import Context
from tinygrad.renderer.amd.sqtt import decode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, VALUINST, InstOp
from tinygrad.runtime.autogen.amd.rdna3.ins import *

def _run_kernel(instructions: list, lx=1, ly=1, lz=1, gx=1, gy=1, gz=1, args_ptr=0) -> bytes:
  """Assemble instructions, run on emulator with PROFILE=1, return the SQTT blob."""
  from test.mockgpu.amd.emu import run_asm, sqtt_traces
  code = b''.join(inst.to_bytes() for inst in instructions)
  buf = (ctypes.c_char * len(code))(*code)
  lib = ctypes.addressof(buf)
  sqtt_traces.clear()
  with Context(PROFILE=1):
    run_asm(lib, len(code), gx, gy, gz, lx, ly, lz, args_ptr)
  assert len(sqtt_traces) == 1, f"expected 1 trace, got {len(sqtt_traces)}"
  return sqtt_traces.pop()

class TestSQTTEncoder(unittest.TestCase):

  def test_simple_salu(self):
    """A simple s_mov + s_endpgm kernel emits SALU INST packet."""
    blob = _run_kernel([s_mov_b32(s[0], 42), s_endpgm()])
    packets = list(decode(blob))
    inst_pkts = [p for p in packets if isinstance(p, INST)]
    self.assertEqual(len(inst_pkts), 1)
    self.assertEqual(inst_pkts[0].op, InstOp.SALU)

  def test_valu_emits_valuinst(self):
    """Regular VALU ops emit VALUINST packets."""
    blob = _run_kernel([v_mov_b32_e32(v[0], 0), v_add_f32_e32(v[1], v[0], v[0]), s_endpgm()])
    packets = list(decode(blob))
    valu_pkts = [p for p in packets if isinstance(p, VALUINST)]
    self.assertEqual(len(valu_pkts), 2)
    # no INST packets for regular VALU
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 0)

  def test_waitcnt_emits_immediate(self):
    """s_waitcnt and s_nop emit IMMEDIATE packets."""
    blob = _run_kernel([s_nop(simm16=0), s_waitcnt(simm16=0), s_endpgm()])
    imm_pkts = [p for p in decode(blob) if isinstance(p, IMMEDIATE)]
    self.assertEqual(len(imm_pkts), 2)  # s_nop + s_waitcnt

  def test_endpgm_skipped(self):
    """s_endpgm does not emit any packet."""
    blob = _run_kernel([s_endpgm()])
    packets = list(decode(blob))
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 0)
    self.assertEqual(len([p for p in packets if isinstance(p, IMMEDIATE)]), 0)

  def test_wave_lifecycle(self):
    """Every WAVESTART has a matching WAVEEND."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_endpgm()])
    packets = list(decode(blob))
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVESTART)), sum(1 for p in packets if isinstance(p, WAVEEND)))

  def test_layout_header(self):
    """First packet is LAYOUT_HEADER with layout=3."""
    blob = _run_kernel([s_endpgm()])
    packets = list(decode(blob))
    self.assertIsInstance(packets[0], LAYOUT_HEADER)
    self.assertEqual(packets[0].layout, 3)

  def test_blob_32byte_aligned(self):
    """SQTT blob is 32-byte aligned."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_mov_b32(s[1], 1), s_endpgm()])
    self.assertEqual(len(blob) % 32, 0)

  def test_multiple_waves(self):
    """Multiple wavefronts each get their own WAVESTART/WAVEEND."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_endpgm()], lx=64)  # 64 threads = 2 waves (WAVE_SIZE=32)
    packets = list(decode(blob))
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVESTART)), 2)
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVEEND)), 2)

  def test_branch_taken_and_not_taken(self):
    """A loop with s_cbranch_scc1 emits JUMP when taken, JUMP_NO on final iteration."""
    # s[0] = 2; loop: s[0] -= 1; cmp s[0] != 0 (SCC=1 if true); cbranch_scc1 loop; endpgm
    # iteration 1: s[0]=2→1, SCC=1 (1!=0), branch taken (JUMP)
    # iteration 2: s[0]=1→0, SCC=0 (0==0), branch not taken (JUMP_NO)
    blob = _run_kernel([s_mov_b32(s[0], 2), s_sub_u32(s[0], s[0], 1), s_cmp_lg_u32(s[0], 0), s_cbranch_scc1(simm16=-3), s_endpgm()])
    inst_pkts = [p for p in decode(blob) if isinstance(p, INST)]
    ops = [p.op for p in inst_pkts]
    self.assertIn(InstOp.JUMP, ops)
    self.assertIn(InstOp.JUMP_NO, ops)

  def test_timestamps_monotonic(self):
    """Timestamps are monotonically non-decreasing."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_mov_b32(s[1], 1), s_mov_b32(s[2], 2), s_endpgm()])
    times = [p._time for p in decode(blob)]
    self.assertEqual(times, sorted(times))

  def test_no_trace_without_profile(self):
    """No SQTT trace is emitted when PROFILE=0."""
    from test.mockgpu.amd.emu import run_asm, sqtt_traces
    code = s_endpgm().to_bytes()
    buf = (ctypes.c_char * len(code))(*code)
    sqtt_traces.clear()
    with Context(PROFILE=0):
      run_asm(ctypes.addressof(buf), len(code), 1, 1, 1, 1, 1, 1, 0)
    self.assertEqual(len(sqtt_traces), 0)

if __name__ == "__main__":
  unittest.main()
