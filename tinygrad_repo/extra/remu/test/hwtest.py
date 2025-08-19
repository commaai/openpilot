import numpy as np
import unittest
import subprocess, struct, math
from typing import cast
from tinygrad.runtime.ops_amd import AMDProgram, AMDDevice
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import diskcache, OSX, getenv

@diskcache
def assemble(code:str) -> bytes:
  try:
    LLVM_MC = "llvm-mc" if OSX else "/opt/rocm/llvm/bin/llvm-mc"
    return subprocess.run([LLVM_MC, "--arch=amdgcn", "--mcpu=gfx1100", "--triple=amdgcn-amd-amdhsa", "-filetype=obj", "-o", "-"],
                           input=code.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
  except subprocess.CalledProcessError as e:
    print("stderr:")
    print(e.stderr.decode())
    raise

# copied from extra/rdna
def get_prg(code:str, v_cnt:int, s_cnt:int):
  function_name = "test"
  metadata = f"""
amdhsa.kernels:
- .args:
  - .address_space: global
    .name: buf_0
    .offset: 0
    .size: 8
    .type_name: unsigned int*
    .value_kind: global_buffer
  .group_segment_fixed_size: 0
  .kernarg_segment_align: 8
  .kernarg_segment_size: 8
  .language: OpenCL C
  .language_version:
  - 1
  - 2
  .max_flat_workgroup_size: 256
  .name: test
  .private_segment_fixed_size: 0
  .sgpr_count: {s_cnt}
  .sgpr_spill_count: 0
  .symbol: test.kd
  .uses_dynamic_stack: false
  .vgpr_count: {v_cnt}
  .vgpr_spill_count: 0
  .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
- 1
- 2
  """
  boilerplate_start = f"""
  .rodata
  .global {function_name}.kd
  .type {function_name}.kd,STT_OBJECT
  .align 0x10
  .amdhsa_kernel {function_name}"""
  kernel_desc = {
    '.amdhsa_group_segment_fixed_size': 0, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': 0,
    '.amdhsa_next_free_vgpr': v_cnt,   # this matters!
    '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
    '.amdhsa_next_free_sgpr': s_cnt,
    '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0, '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3,
    '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1, '.amdhsa_fp16_overflow': 0,
    '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0, '.amdhsa_enable_private_segment': 0,
    '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1,
    '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 2, # is amdhsa_system_vgpr_workitem_id real?
    '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0,
    '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0, '.amdhsa_exception_fp_ieee_underflow': 0,
    '.amdhsa_exception_fp_ieee_inexact': 0, '.amdhsa_exception_int_div_zero': 0,
    '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0, '.amdhsa_user_sgpr_kernarg_segment_ptr': 1,
    '.amdhsa_user_sgpr_dispatch_id': 0, '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0}
  code_start = f""".end_amdhsa_kernel
  .text
  .global {function_name}
  .type {function_name},@function
  .p2align 8
  {function_name}:
  """
  ret = ".amdgpu_metadata\n" + metadata + ".end_amdgpu_metadata" + boilerplate_start + "\n" + '\n'.join("%s %d" % x for x in kernel_desc.items()) \
      + "\n" + code_start + code + f"\n.size {function_name}, .-{function_name}"
  return AMDProgram(cast(AMDDevice, Device["AMD"]), function_name, assemble(ret))

def get_output(s:str, n_threads:int=1):
  assert n_threads <= 32
  code = "\n".join(["s_load_b64 s[0:1], s[0:1], null", "v_lshlrev_b32_e32 v0, 2, v0", s,
    "s_waitcnt 0",
    "global_store_b32 v0, v1, s[0:1]",
    "s_nop 0", "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", "s_endpgm"])
  test = Tensor.zeros((n_threads,), dtype=dtypes.uint32).contiguous().realize().uop.buffer
  prg = get_prg(code, 32, 32)
  prg(test._buf, global_size=(1, 1, 1), local_size=(n_threads, 1, 1), wait=True)
  return test.numpy()

def f16_to_bits(x:float) -> int: return struct.unpack('<H', struct.pack('<e', x))[0]
def f32_from_bits(x:int) -> float: return struct.unpack('<f', struct.pack('<I', x))[0]
def f32_to_bits(x:float) -> int: return struct.unpack('<I', struct.pack('<f', x))[0]

@unittest.skipUnless(Device.DEFAULT == "AMD", "tests RDNA3")
class TestHW(unittest.TestCase):
  def setUp(self):
    if getenv("MOCKGPU"): subprocess.run(["cargo", "build", "--release", "--manifest-path", "./extra/remu/Cargo.toml"], check=True)

  def test_simple(self):
    out = get_output("""
    v_mov_b32_e32 v10 42
    v_mov_b32_e32 v1 v10
    """, n_threads=2)
    np.testing.assert_equal(out, 42)

  def test_exec_mov(self):
    out = get_output("""
    v_mov_b32_e32 v10 42
    s_mov_b32_e32 exec_lo 0b10
    v_mov_b32_e32 v10 10
    s_mov_b32_e32 exec_lo 0b11
    v_mov_b32_e32 v1 v10
    """, n_threads=2)
    np.testing.assert_equal(out, [42, 10])

  def test_exec_cmp_vopc(self):
    out = get_output("""
    s_mov_b32 vcc_lo 0 // reset vcc
    v_mov_b32_e32 v10 42
    v_mov_b32_e32 v11 10
    s_mov_b32_e32 exec_lo 0b01
    v_cmp_ne_u32 v10 v11
    s_mov_b32_e32 exec_lo 0b11
    v_mov_b32_e32 v1 vcc_lo
    """, n_threads=2)
    np.testing.assert_equal(out, 0b01)

  def test_exec_cmpx_vop3(self):
    out = get_output("""
    v_mov_b32_e32 v10 42
    v_mov_b32_e32 v11 10
    s_mov_b32_e32 exec_lo 0b01
    v_cmpx_ne_u32 v10 v11
    s_mov_b32_e32 s10 exec_lo
    s_mov_b32_e32 exec_lo 0b11
    v_mov_b32_e32 v1 s10
    """, n_threads=2)
    np.testing.assert_equal(out, 0b01)

  def test_fmac_vop3_modifier(self):
    init_state = f"""
    v_mov_b32_e32 v10 {f16_to_bits(4.0)}
    v_mov_b32_e32 v11 {f16_to_bits(3.0)}
    v_mov_b32_e32 v1 {f16_to_bits(2.0)}
    """
    self.assertEqual(get_output(init_state+"\n"+"v_fmac_f16_e64 v1 v11 v10"), f16_to_bits(14.))
    self.assertEqual(get_output(init_state+"\n"+"v_fmac_f16_e64 v1 -v11 v10"), f16_to_bits(-10.))
    self.assertEqual(get_output(init_state+"\n"+"v_fmac_f16_e64 v1 -v11 -v10"), f16_to_bits(14.))

  def test_s_abs_i32(self):
    def s_abs_i32(x, y, dst="s10", scc=0):
      for reg,val in [(dst, y), ("scc", scc)]:
        self.assertEqual(get_output(f"""
        s_mov_b32_e32 {dst} {x}
        s_abs_i32 {dst} {dst}
        v_mov_b32_e32 v1 {reg}
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
      v_mov_b32_e32 v1 {f32_to_bits(x)}
      v_rcp_f32_e64 v1, -v1
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
      out = get_output(f"""
      v_mov_b32_e32 v1 {f32_to_bits(x)}
      s_mov_b32_e32 s10 1 // always pick -v1
      v_cndmask_b32 v1, v1, -v1 s10
      """)[0]
      assert out == f32_to_bits(y), f"{f32_from_bits(out)} != {y} / {out} != {f32_to_bits(y)}"
    v_neg(-0.0, 0.0)
    v_neg(0.0, -0.0)
    v_neg(2.0, -2.0)
    v_neg(math.inf, -math.inf)
    v_neg(-math.inf, math.inf)

if __name__ == "__main__":
  unittest.main()
