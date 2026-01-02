# ruff: noqa: F405, F403
# allow define from star imports

import unittest
import textwrap

from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp, Ops, track_rewrites
from tinygrad.renderer import ProgramSpec
from tinygrad.helpers import TracingKey, getenv
from tinygrad.engine.realize import ExecItem, CompiledRunner

from extra.assembly.rdna3.autogen import *

# TODO: use the RDNA3 renderer when it's in master
template = """.text
.globl fn_name
.p2align 8
.type fn_name,@function
fn_name:
  INSTRUCTION

.rodata
.p2align 6
.amdhsa_kernel fn_name
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: fn_name
    .symbol: fn_name.kd
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 8
    .max_flat_workgroup_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .args:
      - .address_space:  global
        .name:           a
        .offset:         0
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
...
.end_amdgpu_metadata
"""

@track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, ret=ret))
def run_asm(name:str, insts:list) -> ProgramSpec:
  src = "\n".join([inst if isinstance(inst, str) else inst.disasm() for inst in insts])
  prg = ProgramSpec(name, src:=template.replace("fn_name", name).replace("INSTRUCTION", textwrap.dedent(src)), Device.DEFAULT, UOp(Ops.SINK),
                    lib=Device[Device.DEFAULT].compiler.compile(src), global_size=[1, 1, 1], local_size=[1, 1, 1], globals=[0])
  ei = ExecItem(UOp(Ops.SINK), [Tensor.empty(1).uop.buffer.ensure_allocated()], prg=CompiledRunner(prg))
  ei.run()
  return prg

@unittest.skipUnless(Device.DEFAULT == "AMD" and not getenv("AMD_LLVM"), "only on AMD with comgr")
class TestCfg(unittest.TestCase):
  def setUp(self):
    arch = Device["AMD"].arch
    if not any(arch.startswith(a) for a in {"gfx11", "gfx12"}):
      self.skipTest(f"tests written for RDNA, got arch {arch}")

  def test_simple(self):
    run_asm("simple", [
      "entry:",
        s_branch("bb1"),
      "bb1:",
        s_endpgm(),
    ])

  def test_diamond(self):
    run_asm("diamond", [
      "entry:",
        s_cmp_eq_i32(s[0], 0),
        s_cbranch_scc1("if"),
        s_branch("else"),
      "if:",
        s_nop(1),
        s_branch("end"),
      "else:",
        s_nop(0),
      "end:",
        s_endpgm(),
    ])

  def test_loop(self):
    run_asm("simple_loop", [
      "entry:",
        s_mov_b32(s[1], 4),
      "loop:",
        s_add_u32(s[1], s[1], -1),
        s_cmp_eq_i32(s[1], 0),
        s_cbranch_scc0("loop"),
        s_endpgm(),
    ])

  def test_loop_branch(self):
    run_asm("loop_if", [
      "entry:",
        s_mov_b32(s[1], 4),
      "loop:",
        s_add_u32(s[1], s[1], -1),
        s_cmp_eq_i32(s[1], 2),
        s_cbranch_scc1("cond"),
        s_branch("cont"),
      "cond:",
        s_add_u32(s[1], s[1], -2),
      "cont:",
        s_cmp_eq_i32(s[1], 0),
        s_cbranch_scc0("loop"),
        s_endpgm(),
    ])

  def test_loop_break(self):
    run_asm("loop_break", [
      "entry:",
        s_mov_b32(s[1], 8),
      "loop:",
        s_add_u32(s[1], s[1], -1),
        s_cmp_eq_i32(s[1], 5),
        s_cbranch_scc1("break"),
        s_cmp_eq_i32(s[1], 0),
        s_cbranch_scc0("loop"),
      "break:",
        s_endpgm(),
    ])

  def test_switch(self):
    run_asm("switch_case", [
      "entry:",
        s_cmp_eq_i32(s[0], 0),
        s_cbranch_scc1("case0"),
        s_cmp_eq_i32(s[0], 1),
        s_cbranch_scc1("case1"),
        s_branch("case2"),
      "case0:",
        s_nop(0),
        s_branch("join"),
      "case1:",
        s_nop(1),
        s_branch("join"),
      "case2:",
        s_nop(2),
        s_branch("join"),
      "join:",
        s_endpgm(),
    ])

  def test_ping_pong(self):
    run_asm("ping_pong", [
      "entry:",
        s_cmp_eq_i32(s[0], 0),
        s_cbranch_scc1("ping"),
        s_branch("pong"),
      "ping:",
        s_cmp_eq_i32(s[1], 0),
        s_cbranch_scc1("pong"),
        s_branch("end"),
      "pong:",
        s_cmp_eq_i32(s[2], 0),
        s_cbranch_scc1("ping"),
      "end:",
        s_endpgm(),
    ])

if __name__ == "__main__":
  unittest.main()
