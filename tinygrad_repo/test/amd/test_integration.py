#!/usr/bin/env python3
"""Integration test: round-trip RDNA3 assembly through LLVM toolchain."""
import unittest
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.amd.helpers import llvm_assemble, llvm_disasm

def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)

def assemble_and_disassemble(instructions: list, mcpu: str = "gfx1100", mattr: str = "+real-true16,+wavefrontsize32") -> list[str]:
  """Assemble instructions with our DSL, then disassemble with LLVM."""
  code_bytes = b''.join(inst.to_bytes() for inst in instructions)
  return llvm_disasm(code_bytes, mcpu, mattr)

class TestIntegration(unittest.TestCase):
  """Test our DSL output matches LLVM disassembly."""

  def test_simple_sop1(self):
    """Test SOP1 instructions round-trip."""
    instructions = [s_mov_b32(s[0], s[1]), s_mov_b32(s[2], 0), s_not_b32(s[3], s[4])]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_mov_b32', disasm[0])
    self.assertIn('s_mov_b32', disasm[1])
    self.assertIn('s_not_b32', disasm[2])

  def test_simple_sop2(self):
    """Test SOP2 instructions round-trip."""
    instructions = [s_add_u32(s[0], s[1], s[2]), s_sub_u32(s[3], s[4], 10), s_and_b32(s[5], s[6], s[7])]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_add_u32', disasm[0])
    self.assertIn('s_sub_u32', disasm[1])
    self.assertIn('s_and_b32', disasm[2])

  def test_simple_vop2(self):
    """Test VOP2 instructions round-trip."""
    instructions = [v_add_f32_e32(v[0], v[1], v[2]), v_mul_f32_e32(v[3], 1.0, v[4]), v_and_b32_e32(v[5], 10, v[6])]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('v_add_f32', disasm[0])
    self.assertIn('v_mul_f32', disasm[1])

  def test_control_flow(self):
    """Test control flow instructions."""
    instructions = [s_waitcnt(simm16=waitcnt(lgkmcnt=0)), s_endpgm()]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_waitcnt', disasm[0])
    self.assertIn('s_endpgm', disasm[1])

  def test_memory_ops(self):
    """Test memory instructions."""
    instructions = [s_load_b32(s[0], s[0:1], NULL), s_waitcnt(simm16=waitcnt(lgkmcnt=0)), global_store_b32(addr=v[0:1], data=v[2], saddr=OFF),
                    s_endpgm()]
    disasm = assemble_and_disassemble(instructions)
    self.assertIn('s_load_b32', disasm[0])
    self.assertIn('s_waitcnt', disasm[1])
    self.assertIn('global_store_b32', disasm[2])

  def test_full_kernel(self):
    """Test a complete kernel similar to tinygrad output."""
    instructions = [v_mov_b32_e32(v[0], s[0]), v_mov_b32_e32(v[1], s[1]), global_load_b32(vdst=v[2], addr=v[0:1], saddr=OFF),
                    s_waitcnt(simm16=waitcnt(vmcnt=0)), v_add_f32_e32(v[2], 1.0, v[2]), global_store_b32(addr=v[0:1], data=v[2], saddr=OFF),
                    s_endpgm()]
    disasm = assemble_and_disassemble(instructions)
    self.assertTrue(any('global_load' in d for d in disasm))
    self.assertTrue(any('v_add_f32' in d for d in disasm))
    self.assertTrue(any('global_store' in d for d in disasm))
    self.assertTrue(any('s_endpgm' in d for d in disasm))

  def test_bytes_roundtrip(self):
    """Test that our bytes match what LLVM assembler produces."""
    inst = s_mov_b32(s[0], s[1])
    our_bytes = inst.to_bytes()
    llvm_bytes = llvm_assemble(["s_mov_b32 s0, s1"], "gfx1100", "+real-true16,+wavefrontsize32")[0]
    self.assertEqual(our_bytes, llvm_bytes, f"Bytes mismatch: ours={our_bytes.hex()} LLVM={llvm_bytes.hex()}")

class TestTinygradIntegration(unittest.TestCase):
  """Test that we can parse tinygrad kernel disassembly."""

  def _get_kernel_code(self, op_fn) -> bytes:
    from tinygrad import Tensor
    from tinygrad.helpers import Target
    from tinygrad.codegen import to_program
    from tinygrad.renderer.llvmir import AMDLLVMRenderer
    from tinygrad.runtime.support.elf import elf_loader
    from tinygrad.uop.ops import Ops

    result = op_fn(Tensor)
    linear = result.schedule_linear()
    sink_items = [call for call in linear.src if call.src[0].op == Ops.SINK]
    assert len(sink_items) > 0, "No SINK in schedule"
    renderer = AMDLLVMRenderer(Target("AMD", arch='gfx1100'))
    prg = to_program(sink_items[0].src[0], renderer)
    lib = renderer.compiler.compile(prg.src[3].arg)
    return next(s.content for s in elf_loader(lib)[1] if s.name == ".text")

  def test_simple_add_kernel(self):
    """Generate a simple add kernel from tinygrad and verify disassembly."""
    code = self._get_kernel_code(lambda T: T([1.0, 2.0, 3.0, 4.0]).realize() + T([5.0, 6.0, 7.0, 8.0]).realize())
    instrs = llvm_disasm(code, "gfx1100", "+real-true16,+wavefrontsize32")
    self.assertTrue(len(instrs) > 0, "No instructions in disassembly")
    self.assertTrue(any('s_endpgm' in i for i in instrs), "Missing s_endpgm")

  def test_matmul_kernel(self):
    """Generate a matmul kernel and verify disassembly has expected patterns."""
    code = self._get_kernel_code(lambda T: T.rand(4, 4).realize() @ T.rand(4, 4).realize())
    instrs = llvm_disasm(code, "gfx1100", "+real-true16,+wavefrontsize32")
    has_mul = any('mul' in i.lower() for i in instrs)
    has_add = any('add' in i.lower() for i in instrs)
    self.assertTrue(has_mul or has_add, "Matmul should have mul/add ops")

  def test_disasm_to_bytes_roundtrip(self):
    """Verify s_endpgm encoding matches between our DSL and LLVM."""
    our_bytes = s_endpgm().to_bytes()
    llvm_bytes = llvm_assemble(["s_endpgm"], "gfx1100", "+real-true16,+wavefrontsize32")[0]
    self.assertEqual(our_bytes, llvm_bytes, f"s_endpgm mismatch: ours={our_bytes.hex()} LLVM={llvm_bytes.hex()}")

if __name__ == "__main__":
  unittest.main()
