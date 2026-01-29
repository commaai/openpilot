# ruff: noqa: F405, F403
# allow define from star imports

import numpy as np
import unittest
import subprocess, struct, math, functools
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv

from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.asm import waitcnt

from test.testextra.test_cfg_viz import asm_kernel

def get_output(asm:list, n_threads:int=1, vdst:VGPR=v[1]):
  out = Tensor([0]*n_threads, dtype=dtypes.uint32).realize()
  insts = [
    s_load_b64(s[0:1], s[0:1], NULL),
    *asm,
    v_lshlrev_b32_e32(v[0], 2, v[0]),
    s_waitcnt(simm16=waitcnt(lgkmcnt=0)),
    #global_store_b32(v[0], v[1], s[0:1]),
    global_store_b32(addr=v[0], data=vdst, saddr=s[0:1]),
    s_endpgm()
  ]
  out = Tensor.custom_kernel(out, fxn=functools.partial(asm_kernel, name="test", insts=insts, device=out.device, n_threads=n_threads))[0]
  out.realize()
  return out.tolist()

def f16_to_bits(x:float) -> int: return struct.unpack('<H', struct.pack('<e', x))[0]
def f32_from_bits(x:int) -> float: return struct.unpack('<f', struct.pack('<I', x))[0]
def f32_to_bits(x:float) -> int: return struct.unpack('<I', struct.pack('<f', x))[0]

@unittest.skipUnless(Device.DEFAULT == "AMD", "tests RDNA3")
class TestHW(unittest.TestCase):
  def setUp(self):
    if getenv("MOCKGPU"): subprocess.run(["cargo", "build", "--release", "--manifest-path", "./extra/remu/Cargo.toml"], check=True)

  def test_simple_v_mov(self):
    out = get_output([
      v_mov_b32_e32(v[1], 2),
    ])
    self.assertEqual(out, [2])

  def test_simple_s_mov(self):
    out = get_output([
      s_mov_b32(s[7], 0x7fffffff),
      v_mov_b32_e32(v[1], s[7]),
    ])
    self.assertEqual(out, [0x7fffffff])

  def test_exec_mov(self):
    out = get_output([
      v_mov_b32_e32(v[1], 42),
      s_mov_b32(EXEC_LO, 0b10),
      v_mov_b32_e32(v[1], 10),
      s_mov_b32(EXEC_LO, 0b11),
    ], n_threads=2)
    np.testing.assert_equal(out, [42, 10])

  def test_exec_cmp_vopc(self):
    out = get_output([
      s_mov_b32(VCC_LO, 0), # reset vcc
      v_mov_b32_e32(v[1], 42),
      v_mov_b32_e32(v[2], 10),
      s_mov_b32(EXEC_LO, 0b01),
      v_cmp_ne_u32_e32(v[1], v[2]),
      s_mov_b32(EXEC_LO, 0b11),
      v_mov_b32_e32(v[1], VCC_LO),
    ], n_threads=2)[0]
    np.testing.assert_equal(out, 1)

  def test_exec_cmpx_vop3(self):
    out = get_output([
      s_mov_b32(EXEC_LO, 0b11),
      v_mov_b32_e32(v[1], 42),
      v_mov_b32_e32(v[2], 10),
      s_mov_b32(EXEC_LO, 0b01),
      v_cmpx_ne_u32_e32(v[1], v[2]),
      s_mov_b32(s[10], EXEC_LO),
      s_mov_b32(EXEC_LO, 0b11),
      v_mov_b32_e32(v[1], s[10]),
    ], n_threads=2)[0]
    np.testing.assert_equal(out & 0b11, 0b01)

  def test_fmac_vop3_modifier(self):
    init_state = [
      v_mov_b32_e32(a:=v[1], f16_to_bits(4.0)),
      v_mov_b32_e32(b:=v[2], f16_to_bits(3.0)),
      v_mov_b32_e32(c:=v[3], f16_to_bits(2.0)),
    ]
    def run_fmac(a, b): return get_output(init_state+[v_fmac_f16_e64(c, a, b)], vdst=c)[0]
    self.assertEqual(run_fmac(a, b), f16_to_bits(14.0))
    self.assertEqual(run_fmac(a, -b), f16_to_bits(-10.0))
    self.assertEqual(run_fmac(-a, -b), f16_to_bits(14.0))

  def test_s_abs_i32(self):
    def check(x, y, dst=s[10], scc=0):
      for reg,val in [(dst, y), (SCC, scc)]:
        self.assertEqual(get_output([
          s_mov_b32(dst, x),
          s_abs_i32(dst, dst),
          v_mov_b32_e32(v[1], reg)
        ])[0], val)

    check(0x00000001, 0x00000001, scc=1)
    check(0x7fffffff, 0x7fffffff, scc=1)
    check(0x80000000, 0x80000000, scc=1)
    check(0x80000001, 0x7fffffff, scc=1)
    check(0x80000002, 0x7ffffffe, scc=1)
    check(0xffffffff, 0x00000001, scc=1)
    check(0, 0, scc=0)

  def test_v_rcp_f32_neg_vop3(self):
    def v_neg_rcp_f32(x:float, y:float):
      out = get_output([
        v_mov_b32_e32(v[2], f32_to_bits(x)),
        v_rcp_f32_e64(v[2], -v[2]),
      ], vdst=v[2])[0]
      assert out == f32_to_bits(y), f"{f32_from_bits(out)} != {y} / {out} != {f32_to_bits(y)}"

    v_neg_rcp_f32(math.inf, -0.0)
    v_neg_rcp_f32(-math.inf, 0.0)
    v_neg_rcp_f32(0.0, -math.inf)
    v_neg_rcp_f32(-0.0, math.inf)
    v_neg_rcp_f32(-2.0, 0.5)
    v_neg_rcp_f32(2.0, -0.5)

  def test_v_cndmask_b32_neg(self):
    def v_neg(x:float, y:float):
      out = get_output([
        v_mov_b32_e32(v[1], f32_to_bits(x)),
        s_mov_b32(s[10], 1),
        v_cndmask_b32_e64(v[1], v[1], -v[1], s[10]),
      ])[0]
      assert out == f32_to_bits(y), f"{f32_from_bits(out)} != {y} / {out} != {f32_to_bits(y)}"

    v_neg(-0.0, 0.0)
    v_neg(0.0, -0.0)
    v_neg(2.0, -2.0)
    v_neg(math.inf, -math.inf)
    v_neg(-math.inf, math.inf)

  @unittest.skip("how does VOPD work in the dsl")
  def test_v_subrev_wrap(self):
    out = get_output([
      #v_dual_mov_b32(v[1], 0xffffffff, v[2], 0x0),
      #v_dual_mov_b32(vdstx=v[1], srcx=0xffffffff, vdsty=v[2], srcy=0x0),
      #VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32, vdstx=v[1], srcx=0xffffffff, vdsty=v[2], srcy=0x0),
      v_subrev_co_u32(v[2], VCC_LO, v[2], v[1]),
    ], vdst=v[2])[0]
    self.assertEqual(out, 0xffff_ffff)

if __name__ == "__main__":
  unittest.main()
