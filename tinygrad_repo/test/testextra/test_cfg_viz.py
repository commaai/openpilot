# ruff: noqa: F405, F403
# allow define from star imports

import unittest
import textwrap, functools

from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv
from tinygrad.device import Compiler
from tinygrad.viz.serve import amdgpu_cfg

from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import Inst

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

# TODO: shouldn't need compiler once we can output ELF
# outputs a text disassembly for humans and a machine readable binary
def assemble(name:str, insts:list[str|Inst], compiler:Compiler) -> tuple[str, bytes]:
  asm = "\n".join([inst if isinstance(inst, str) else inst.disasm() for inst in insts])
  src = template.replace("fn_name", name).replace("INSTRUCTION", textwrap.dedent(asm))
  return (src, compiler.compile(src))

def asm_kernel(out:UOp, insts:list[str|Inst], name:str, device:str, compiler:Compiler, n_threads:int=1, n_workgroups:int=1) -> UOp:
  lidx = UOp.special(n_threads, "lidx0")
  gidx = UOp.special(n_workgroups, "gidx0")
  sink = UOp.sink(out, lidx, gidx, arg=KernelInfo(name=name))
  src, lib = assemble(name, insts, compiler)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def run_asm(name:str, insts:list) -> None:
  fxn = functools.partial(asm_kernel, insts=insts, name=name, device=Device.DEFAULT, compiler=Device[Device.DEFAULT].compiler)
  out = Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0]
  out.realize()

@unittest.skipUnless(Device.DEFAULT == "AMD" and not getenv("AMD_LLVM"), "only on AMD with comgr")
class TestCfg(unittest.TestCase):
  def setUp(self):
    arch = Device["AMD"].arch
    if not any(arch.startswith(a) for a in {"gfx11", "gfx12"}):
      self.skipTest(f"tests written for RDNA, got arch {arch}")

  def test_simple(self):
    run_asm("simple", [
      "entry:",
        "s_branch bb1",
      "bb1:",
        s_endpgm(),
        s_code_end(),
    ])

  def test_diamond(self):
    run_asm("diamond", insts:=[
      "entry:",
        s_mov_b32(s[0], 0),
        s_mov_b32(s[1], 0),
        s_cmp_eq_u64(s[0:1], 0),
        "s_cbranch_scc1 if",
        "s_branch else",
      "if:",
        s_nop(1),
        "s_branch end",
      "else:",
        s_nop(0),
      "end:",
        s_endpgm(),
        s_code_end(),
    ])
    _, lib = assemble("diamond", insts, Device[Device.DEFAULT].compiler)
    cfg = amdgpu_cfg(lib, Device[Device.DEFAULT].device_props()["gfx_target_version"])["data"]
    self.assertEqual(len(cfg["blocks"]), 5)
    edge_count = sum(len(v) for v in cfg["paths"].values())
    self.assertEqual(edge_count, 5)
    references:dict[str, list[str]] = {}
    for pc, tokens in cfg["pc_tokens"].items():
      for t in tokens:
        for key in t["keys"]: references.setdefault(key, []).append(pc)
    self.assertEqual(len(references["r0"]), 2)
    insts = [cfg["pc_tokens"][pc][0]["st"] for pc in references["r0"]]
    self.assertEqual(insts, ['s_mov_b32', 's_cmp_eq_u64'])

  def test_loop(self):
    run_asm("simple_loop", [
      "entry:",
        s_mov_b32(s[1], 4),
      "loop:",
        s_add_u32(s[1], s[1], -1),
        s_cmp_eq_i32(s[1], 0),
        "s_cbranch_scc0 loop",
        s_endpgm(),
        s_code_end(),
    ])

  def test_loop_branch(self):
    run_asm("loop_if", [
      "entry:",
        s_mov_b32(s[1], 4),
      "loop:",
        s_add_u32(s[1], s[1], -1),
        s_cmp_eq_i32(s[1], 2),
        "s_cbranch_scc1 cond",
        "s_branch cont",
      "cond:",
        s_add_u32(s[1], s[1], -2),
      "cont:",
        s_cmp_eq_i32(s[1], 0),
        "s_cbranch_scc0 loop",
        s_endpgm(),
        s_code_end(),
    ])

  def test_loop_break(self):
    run_asm("loop_break", [
      "entry:",
        s_mov_b32(s[1], 8),
      "loop:",
        s_add_u32(s[1], s[1], -1),
        s_cmp_eq_i32(s[1], 5),
        "s_cbranch_scc1 break",
        s_cmp_eq_i32(s[1], 0),
        "s_cbranch_scc0 loop",
      "break:",
        s_endpgm(),
        s_code_end(),
    ])

  def test_switch(self):
    run_asm("switch_case", [
      "entry:",
        s_cmp_eq_i32(s[0], 0),
        "s_cbranch_scc1 case0",
        s_cmp_eq_i32(s[0], 1),
        "s_cbranch_scc1 case1",
        "s_branch case2",
      "case0:",
        s_nop(0),
        "s_branch join",
      "case1:",
        s_nop(1),
        "s_branch join",
      "case2:",
        s_nop(2),
        "s_branch join",
      "join:",
        s_endpgm(),
        s_code_end(),
    ])

  def test_ping_pong(self):
    run_asm("ping_pong", [
      "entry:",
        s_cmp_eq_i32(s[0], 0),
        "s_cbranch_scc1 ping",
        "s_branch pong",
      "ping:",
        s_cmp_eq_i32(s[1], 0),
        "s_cbranch_scc1 pong",
        "s_branch end",
      "pong:",
        s_cmp_eq_i32(s[2], 0),
        "s_cbranch_scc1 ping",
      "end:",
        s_endpgm(),
        s_code_end(),
    ])

  def test_colored_blocks(self):
    N = 10
    asm = ["entry:", "s_branch init0"]
    for i in range(N):
      asm += [f"init{i}:", s_mov_b32(s[1], i + 1), f"s_branch {(loop:=f'loop{i}')}"]
      asm += [
        f"{loop}:",
          s_nop(i & 7),
          s_add_u32(s[1], s[1], -1),
          s_cmp_eq_i32(s[1], 0),
          f"s_cbranch_scc0 {loop}",
          f"s_branch {'init' + str(i+1) if i + 1 < N else 'end'}",
      ]
    asm += ["end:", s_endpgm(), s_code_end()]
    run_asm("test_colored_blocks", asm)

  def test_jump_back_to_end(self):
    run_asm("jump_back_to_end", [
      "entry:",
        s_mov_b32(s[1], 2),
        "s_cbranch_execz loop",
      "end:",
        s_endpgm(),
      "loop:",
        s_add_u32(s[1], s[1], -1),
        s_cmp_eq_i32(s[1], 0),
        "s_branch end",
        s_code_end(),
    ])

if __name__ == "__main__":
  unittest.main()
