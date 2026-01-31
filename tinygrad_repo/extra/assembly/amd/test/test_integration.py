#!/usr/bin/env python3
"""Integration test: round-trip RDNA3 assembly through AMD toolchain."""
import unittest, io, sys
from extra.assembly.amd.autogen.rdna3.ins import *

def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)

def disassemble(lib: bytes, arch: str = "gfx1100") -> str:
  """Disassemble ELF binary using tinygrad's compiler, return raw output."""
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  old_stdout = sys.stdout
  sys.stdout = io.StringIO()
  HIPCompiler(arch).disassemble(lib)
  output = sys.stdout.getvalue()
  sys.stdout = old_stdout
  return output

def parse_disassembly(raw: str) -> list[str]:
  """Parse disassembly output to list of instruction mnemonics."""
  lines = []
  for line in raw.splitlines():
    if line.startswith('\t'):
      instr = line.split('//')[0].strip()
      if instr: lines.append(instr)
  return lines

def assemble_and_disassemble(instructions: list, arch: str = "gfx1100") -> list[str]:
  """Assemble instructions with our DSL, then disassemble with AMD toolchain."""
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  # Generate bytes from our DSL
  code_bytes = b''.join(inst.to_bytes() for inst in instructions)

  # Wrap in minimal ELF-compatible assembly with .byte directives
  byte_str = ', '.join(f'0x{b:02x}' for b in code_bytes)
  asm_src = f".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\n.byte {byte_str}\n"

  # Assemble with AMD COMGR and disassemble
  lib = HIPCompiler(arch).compile(asm_src)
  return parse_disassembly(disassemble(lib, arch))

class TestIntegration(unittest.TestCase):
  """Test our DSL output matches LLVM disassembly."""

  def test_simple_sop1(self):
    """Test SOP1 instructions round-trip."""
    instructions = [
      s_mov_b32(s[0], s[1]),
      s_mov_b32(s[2], 0),
      s_not_b32(s[3], s[4]),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_mov_b32', disasm[0])
    self.assertIn('s_mov_b32', disasm[1])
    self.assertIn('s_not_b32', disasm[2])

  def test_simple_sop2(self):
    """Test SOP2 instructions round-trip."""
    instructions = [
      s_add_u32(s[0], s[1], s[2]),
      s_sub_u32(s[3], s[4], 10),
      s_and_b32(s[5], s[6], s[7]),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_add_u32', disasm[0])
    self.assertIn('s_sub_u32', disasm[1])
    self.assertIn('s_and_b32', disasm[2])

  def test_simple_vop2(self):
    """Test VOP2 instructions round-trip."""
    instructions = [
      v_add_f32_e32(v[0], v[1], v[2]),
      v_mul_f32_e32(v[3], 1.0, v[4]),  # 1.0 is inline constant
      v_and_b32_e32(v[5], 10, v[6]),  # small inline constant
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('v_add_f32', disasm[0])
    self.assertIn('v_mul_f32', disasm[1])

  def test_control_flow(self):
    """Test control flow instructions."""
    instructions = [
      s_waitcnt(simm16=waitcnt(lgkmcnt=0)),
      s_endpgm(),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_waitcnt', disasm[0])
    self.assertIn('s_endpgm', disasm[1])

  def test_memory_ops(self):
    """Test memory instructions."""
    instructions = [
      s_load_b32(s[0], s[0:1], NULL),
      s_waitcnt(simm16=waitcnt(lgkmcnt=0)),
      global_store_b32(addr=v[0:1], data=v[2], saddr=OFF),
      s_endpgm(),
    ]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_load_b32', disasm[0])
    self.assertIn('s_waitcnt', disasm[1])
    self.assertIn('global_store_b32', disasm[2])

  def test_full_kernel(self):
    """Test a complete kernel similar to tinygrad output."""
    # Simple kernel: load value, add 1, store back
    instructions = [
      # Get thread ID
      v_mov_b32_e32(v[0], s[0]),  # base addr low
      v_mov_b32_e32(v[1], s[1]),  # base addr high
      # Load value
      global_load_b32(vdst=v[2], addr=v[0:1], saddr=OFF),
      s_waitcnt(simm16=waitcnt(vmcnt=0)),
      # Add 1.0
      v_add_f32_e32(v[2], 1.0, v[2]),
      # Store result
      global_store_b32(addr=v[0:1], data=v[2], saddr=OFF),
      s_endpgm(),
    ]
    disasm = assemble_and_disassemble(instructions)
    # Verify key instructions are present
    self.assertTrue(any('global_load' in d for d in disasm))
    self.assertTrue(any('v_add_f32' in d for d in disasm))
    self.assertTrue(any('global_store' in d for d in disasm))
    self.assertTrue(any('s_endpgm' in d for d in disasm))

  def test_bytes_roundtrip(self):
    """Test that our bytes match what AMD assembler produces."""
    from tinygrad.runtime.support.compiler_amd import HIPCompiler

    # Simple instruction
    inst = s_mov_b32(s[0], s[1])
    our_bytes = inst.to_bytes()

    # Assemble same instruction with AMD toolchain
    asm_src = ".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\ns_mov_b32 s0, s1\n"
    compiler = HIPCompiler("gfx1100")
    lib = compiler.compile(asm_src)
    raw = disassemble(lib)

    for line in raw.splitlines():
      if 's_mov_b32' in line and '//' in line:
        # Extract hex bytes from comment: "// 000000001300: BE800001"
        comment = line.split('//')[1].strip()
        hex_str = comment.split(':')[1].strip()
        # Convert big-endian hex string to little-endian bytes
        amd_bytes = bytes.fromhex(hex_str)[::-1]  # reverse for little-endian
        self.assertEqual(our_bytes, amd_bytes, f"Bytes mismatch: ours={our_bytes.hex()} AMD={amd_bytes.hex()}")
        return
    self.fail("Could not find s_mov_b32 in disassembly")

class TestTinygradIntegration(unittest.TestCase):
  """Test that we can parse disassembled tinygrad kernels."""

  def test_simple_add_kernel(self):
    """Generate a simple add kernel from tinygrad and verify disassembly."""
    from tinygrad import Tensor
    from tinygrad.codegen import get_program
    from tinygrad.renderer.cstyle import AMDHIPRenderer
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    from tinygrad.uop.ops import Ops

    # Create a computation that generates a real kernel
    a = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
    b = Tensor([5.0, 6.0, 7.0, 8.0]).realize()
    c = a + b

    # Get schedule and find SINK
    schedule = c.schedule()
    sink_items = [si for si in schedule if si.ast.op == Ops.SINK]
    self.assertTrue(len(sink_items) > 0, "No SINK in schedule")

    # Generate program
    renderer = AMDHIPRenderer('gfx1100')
    prg = get_program(sink_items[0].ast, renderer)
    self.assertIsNotNone(prg.src)

    # Compile and disassemble
    compiler = HIPCompiler('gfx1100')
    lib = compiler.compile(prg.src)
    raw_disasm = disassemble(lib)
    instrs = parse_disassembly(raw_disasm)

    # Verify we got some instructions
    self.assertTrue(len(instrs) > 0, "No instructions in disassembly")
    # Should have an endpgm
    self.assertTrue(any('s_endpgm' in i for i in instrs), "Missing s_endpgm")

  def test_matmul_kernel(self):
    """Generate a matmul kernel and verify disassembly has expected patterns."""
    from tinygrad import Tensor
    from tinygrad.codegen import get_program
    from tinygrad.renderer.cstyle import AMDHIPRenderer
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    from tinygrad.uop.ops import Ops

    # Create a small matmul
    a = Tensor.rand(4, 4).realize()
    b = Tensor.rand(4, 4).realize()
    c = a @ b

    # Get schedule
    schedule = c.schedule()
    sink_items = [si for si in schedule if si.ast.op == Ops.SINK]
    self.assertTrue(len(sink_items) > 0)

    # Generate and compile
    renderer = AMDHIPRenderer('gfx1100')
    prg = get_program(sink_items[0].ast, renderer)
    compiler = HIPCompiler('gfx1100')
    lib = compiler.compile(prg.src)
    raw_disasm = disassemble(lib)
    instrs = parse_disassembly(raw_disasm)

    # Matmul should have multiply and add instructions
    has_mul = any('mul' in i.lower() for i in instrs)
    has_add = any('add' in i.lower() for i in instrs)
    self.assertTrue(has_mul or has_add, "Matmul should have mul/add ops")

  def test_disasm_to_bytes_roundtrip(self):
    """Parse disassembled instructions and verify we can re-encode some of them."""
    from tinygrad import Tensor
    from tinygrad.codegen import get_program
    from tinygrad.renderer.cstyle import AMDHIPRenderer
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    from tinygrad.uop.ops import Ops

    # Simple kernel
    a = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
    b = (a * 2.0)

    schedule = b.schedule()
    sink_items = [si for si in schedule if si.ast.op == Ops.SINK]
    if not sink_items: return  # skip if no kernel

    renderer = AMDHIPRenderer('gfx1100')
    prg = get_program(sink_items[0].ast, renderer)
    compiler = HIPCompiler('gfx1100')
    lib = compiler.compile(prg.src)
    raw_disasm = disassemble(lib)

    # Find s_endpgm and verify we can encode it
    for line in raw_disasm.splitlines():
      if 's_endpgm' in line and '//' in line:
        # Extract bytes from comment
        comment = line.split('//')[1].strip()
        hex_str = comment.split(':')[1].strip()
        amd_bytes = bytes.fromhex(hex_str)[::-1]

        # Our encoding
        our_inst = s_endpgm()
        our_bytes = our_inst.to_bytes()

        self.assertEqual(our_bytes, amd_bytes, f"s_endpgm mismatch: ours={our_bytes.hex()} AMD={amd_bytes.hex()}")
        return

if __name__ == "__main__":
  unittest.main()
