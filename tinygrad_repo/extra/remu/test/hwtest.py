import numpy as np
import unittest
import subprocess, struct, math
from tinygrad import Tensor, dtypes, Device, UOp
from tinygrad.helpers import getenv
from tinygrad.runtime.support.compiler_amd import amdgpu_disassemble
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner

def get_output(asm:str, n_threads:int=1):
  input_asm = "\n".join([ln if ln.strip().startswith('asm volatile') else f'asm volatile("{ln.strip().lstrip()}" : "+v"(a), "+v"(b));'
                         for ln in asm.strip().splitlines() if ln.strip()])
  src = f"""
  typedef long unsigned int size_t;
  extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
  extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, {n_threads}))) test(unsigned int* data0_1) {{
    int l = __ockl_get_local_id(0);
    unsigned a = 0, b = 0, c = 0;
    {input_asm}
    unsigned res;
    asm volatile("v_mov_b32 %0, %1" : "=v"(res) : "v"(a));
    *(data0_1+l) = res;
  }}"""
  t = Tensor.zeros(n_threads, dtype=dtypes.uint32).contiguous().realize()
  prg = ProgramSpec("test", src, Device.DEFAULT, UOp.sink(t), global_size=[1, 1, 1], local_size=[n_threads, 1, 1])
  car = CompiledRunner(prg)
  if getenv("PRINT_ASM"): amdgpu_disassemble(car.lib)
  car([t.uop.buffer], {}, wait=True)
  return t.numpy()

def f16_to_bits(x:float) -> int: return struct.unpack('<H', struct.pack('<e', x))[0]
def f32_from_bits(x:int) -> float: return struct.unpack('<f', struct.pack('<I', x))[0]
def f32_to_bits(x:float) -> int: return struct.unpack('<I', struct.pack('<f', x))[0]

@unittest.skipUnless(Device.DEFAULT == "AMD", "tests RDNA3")
class TestHW(unittest.TestCase):
  def setUp(self):
    if getenv("MOCKGPU"): subprocess.run(["cargo", "build", "--release", "--manifest-path", "./extra/remu/Cargo.toml"], check=True)

  def test_simple(self):
    out = get_output("""
    v_mov_b32_e32 %1 42
    v_mov_b32_e32 %2 %1
    """)[0]
    np.testing.assert_equal(out, 42)

  def test_exec_mov(self):
    out = get_output("""
    v_mov_b32_e32 %1 42
    s_mov_b32_e32 exec_lo 0b10
    v_mov_b32_e32 %1 10
    s_mov_b32_e32 exec_lo 0b11
    v_mov_b32_e32 %2 %1
    """, n_threads=2)
    np.testing.assert_equal(out, [42, 10])

  def test_exec_cmp_vopc(self):
    out = get_output("""
    s_mov_b32 vcc_lo 0 // reset vcc
    v_mov_b32_e32 %1 42
    v_mov_b32_e32 %2 10
    s_mov_b32_e32 exec_lo 0b01
    v_cmp_ne_u32 %1 %2
    s_mov_b32_e32 exec_lo 0b11
    v_mov_b32_e32 %2 vcc_lo
    """, n_threads=2)
    np.testing.assert_equal(out, 0b01)

  def test_exec_cmpx_vop3(self):
    out = get_output("""
    s_mov_b32_e32 exec_lo 0b11
    v_mov_b32_e32 %1 42
    v_mov_b32_e32 %2 10
    s_mov_b32_e32 exec_lo 0b01
    v_cmpx_ne_u32 %1 %2
    s_mov_b32_e32 s10 exec_lo
    s_mov_b32_e32 exec_lo 0b11
    v_mov_b32_e32 %2 s10
    """, n_threads=2)[0]
    np.testing.assert_equal(out & 0b11, 0b01)

  def test_fmac_vop3_modifier(self):
    init_state = f"""
    asm volatile("v_mov_b32_e32 %1, {f16_to_bits(4.0)}" : "+v"(a));
    asm volatile("v_mov_b32_e32 %1, {f16_to_bits(3.0)}" : "+v"(b));
    asm volatile("v_mov_b32_e32 %1, {f16_to_bits(2.0)}" : "+v"(c));
    """
    mov = """asm volatile("v_mov_b32_e32 %1, %2" : "+v"(c), "+v"(a));"""
    def fmac(a, b, c): return f"""asm volatile("v_fmac_f16_e64 {c}, {a}, {b}" : "+v"(c) : "v"(a), "v"(b));"""+"\n"+mov
    self.assertEqual(get_output(init_state+"\n"+fmac("%1", "%2", "%3")), f16_to_bits(14.))
    self.assertEqual(get_output(init_state+"\n"+fmac("%1", "-%2", "%3")), f16_to_bits(-10.))
    self.assertEqual(get_output(init_state+"\n"+fmac("-%1", "-%2", "%3")), f16_to_bits(14.))

  def test_s_abs_i32(self):
    def s_abs_i32(x, y, dst="s10", scc=0):
      for reg,val in [(dst, y), ("scc", scc)]:
        self.assertEqual(get_output(f"""
        s_mov_b32_e32 {dst} {x}
        s_abs_i32 {dst} {dst}
        v_mov_b32_e32 %2 {reg}
        """)[0], val)
    s_abs_i32(0x00000001, 0x00000001, scc=1)
    s_abs_i32(0x7fffffff, 0x7fffffff, scc=1)
    s_abs_i32(0x80000000, 0x80000000, scc=1)
    s_abs_i32(0x80000001, 0x7fffffff, scc=1)
    s_abs_i32(0x80000002, 0x7ffffffe, scc=1)
    s_abs_i32(0xffffffff, 0x00000001, scc=1)
    s_abs_i32(0, 0, scc=0)

  def test_v_rcp_f32_neg_vop3(self):
    def v_neg_rcp_f32(x:float, y:float):
      out = get_output(f"""
      v_mov_b32_e32 %2 {f32_to_bits(x)}
      v_rcp_f32_e64 %2, -%2
      """)[0]
      assert out == f32_to_bits(y), f"{f32_from_bits(out)} != {y} / {out} != {f32_to_bits(y)}"
    v_neg_rcp_f32(math.inf, -0.0)
    v_neg_rcp_f32(-math.inf, 0.0)
    v_neg_rcp_f32(0.0, -math.inf)
    v_neg_rcp_f32(-0.0, math.inf)
    v_neg_rcp_f32(-2.0, 0.5)
    v_neg_rcp_f32(2.0, -0.5)

  def test_v_cndmask_b32_neg(self):
    def v_neg(x:int|float, y:float):
      # always pick -v1
      out = get_output(f"""
      v_mov_b32_e32 %2 {f32_to_bits(x)}
      s_mov_b32_e32 s10 1
      v_cndmask_b32 %2, %2, -%2 s10
      """)[0]
      assert out == f32_to_bits(y), f"{f32_from_bits(out)} != {y} / {out} != {f32_to_bits(y)}"
    v_neg(-0.0, 0.0)
    v_neg(0.0, -0.0)
    v_neg(2.0, -2.0)
    v_neg(math.inf, -math.inf)
    v_neg(-math.inf, math.inf)

  def test_v_subrev_wrap(self):
    out = get_output("""
    v_dual_mov_b32 %1, 0xffffffff :: v_dual_mov_b32 %2, 0x0
    v_subrev_co_u32 %2, vcc_lo, %2, %1
    """)[0]
    self.assertEqual(out, 0xffff_ffff)

if __name__ == "__main__":
  unittest.main()
