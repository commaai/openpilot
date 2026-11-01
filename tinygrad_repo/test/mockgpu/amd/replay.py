#!/usr/bin/env python3
# usage: replay.py [clang,llvm,amd] [asm]
import functools, re, sys
from dataclasses import replace
from test.mockgpu.amd.emu import lift
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.helpers import VERSION, Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer, HIPRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.uop.ops import UOp, Ops
from tinygrad.viz.serve import amd_decode, get_arch, get_cfg, get_elf_section, get_stdout, parse_branch

def get_output(p:UOp, renderer:Renderer, asm:bool=False) -> str:
  nodes = p.toposort(enter_calls=True)
  parts = [x.arg for x in nodes if x.op is Ops.SOURCE]
  for prg in (x for x in nodes if asm and x.op is Ops.PROGRAM):
    binary = next(x.arg for x in prg.src if x.op is Ops.BINARY)
    if renderer.target.device != "AMD":
      disasm = get_stdout(lambda: renderer.compiler.disassemble(binary))
      parts.append(re.sub(r"^.*:\s+file format", "<binary>: file format", disasm, flags=re.MULTILINE))
      continue
    text = get_elf_section(binary, ".text")
    insts = amd_decode(text.content, get_arch(renderer.target.arch), text.header.sh_addr)
    cfg = get_cfg(insts)
    labels = {pc:f"%bb.{i}" for i,pc in enumerate(cfg["blocks"])}
    blocks = []
    for start, pcs in cfg["blocks"].items():
      lines = [f"{labels[start][1:]}:"]
      if successors:=cfg["paths"][start]: lines.append("  successors: " + ", ".join(labels[pc] for pc in successors))
      for pc in pcs:
        op = getattr(inst:=insts[pc], "op_name", "").lower()
        if (offset:=parse_branch(inst, op)) is not None: lines.append(f"  {op} {labels[pc+inst.size()+offset]}")
        else: lines.append(f"  {inst}")
      blocks.append("\n".join(lines))
    parts.append("\n\n".join(blocks))
  return "\n".join(parts)

def replay_asm(ast:UOp, code:bytes, arch:str, renderer:Renderer, dispatch:tuple, name:str,
               backends:tuple[str, ...], asm:bool=False) -> tuple[str, str, tuple]:
  compare = lift(code, arch, renderer, dispatch, name)
  good_outs, compare_outs = [], []
  for backend in backends:
    target = Target("AMD", "HIP", {"rdna3":"gfx1100", "rdna4":"gfx1200", "cdna":"gfx950"}[arch]) if backend == "amd" else \
      replace(Device["CPU"].renderer.target, renderer=backend.upper())
    ren = {"clang":ClangRenderer, "llvm":CPULLVMRenderer, "amd":HIPRenderer}[backend](target)
    good_outs.append(get_output(to_program(ast, ren), ren, asm))
    compare_outs.append(get_output(to_program(compare, ren), ren, asm))
  return "\n".join(good_outs), "\n".join(compare_outs), (name, backends, dispatch)

if __name__ == "__main__":
  from test.external.process_replay.process_replay import main
  backends = tuple((sys.argv[1] if len(sys.argv) > 1 else "clang").split(","))
  main({"lift":functools.partial(replay_asm, backends=backends, asm=bool(sys.argv[2:]))}, f"emu_process_replay_{VERSION}")
