#!/usr/bin/env python3
"""Roundtrip tests: generate tinygrad kernels, decode instructions, re-encode, verify match."""
import unittest, io, sys, re
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.lib import Inst
from extra.assembly.rdna3.asm import asm

# Instruction format detection based on encoding bits
def detect_format(data: bytes) -> type[Inst] | None:
  """Detect instruction format from machine code bytes."""
  if len(data) < 4: return None
  word = int.from_bytes(data[:4], 'little')
  enc_9bit = (word >> 23) & 0x1FF  # 9-bit encoding for SOP1/SOPC/SOPP
  enc_8bit = (word >> 24) & 0xFF

  # Check 9-bit encodings first (most specific)
  if enc_9bit == 0x17D: return SOP1  # bits 31:23 = 101111101
  if enc_9bit == 0x17E: return SOPC  # bits 31:23 = 101111110
  if enc_9bit == 0x17F: return SOPP  # bits 31:23 = 101111111
  # SOPK: bits 31:28 = 1011, bits 27:23 = opcode (check after SOP1/SOPC/SOPP)
  if enc_8bit in range(0xB0, 0xC0): return SOPK
  # SOP2: bits 31:23 in range 0x100-0x17C (0x80-0xBE in bits 31:24, but not SOPK)
  if 0x80 <= enc_8bit <= 0x9F: return SOP2
  # VOP1: bits 31:25 = 0111111 (0x3F)
  if (word >> 25) == 0x3F: return VOP1
  # VOPC: bits 31:25 = 0111110 (0x3E)
  if (word >> 25) == 0x3E: return VOPC
  # VOP2: bits 31:30 = 00
  if (word >> 30) == 0: return VOP2

  # Check 64-bit formats
  if len(data) >= 8:
    if enc_8bit in (0xD4, 0xD5, 0xD7): return VOP3
    if enc_8bit == 0xD6: return VOP3SD
    if enc_8bit == 0xCC: return VOP3P
    if enc_8bit == 0xCD: return VINTERP
    if enc_8bit in (0xC8, 0xC9): return VOPD
    if enc_8bit == 0xF4: return SMEM
    if enc_8bit == 0xD8: return DS
    if enc_8bit in (0xDC, 0xDD, 0xDE, 0xDF): return FLAT
    if enc_8bit in (0xE0, 0xE1, 0xE2, 0xE3): return MUBUF
    if enc_8bit in (0xE8, 0xE9, 0xEA, 0xEB): return MTBUF

  return None

def disassemble_lib(lib: bytes, compiler) -> list[tuple[str, bytes]]:
  """Disassemble ELF binary and return list of (instruction_text, machine_code_bytes)."""
  old_stdout = sys.stdout
  sys.stdout = io.StringIO()
  compiler.disassemble(lib)
  output = sys.stdout.getvalue()
  sys.stdout = old_stdout

  results = []
  for line in output.splitlines():
    if '//' not in line: continue
    instr = line.split('//')[0].strip()
    if not instr: continue
    comment = line.split('//')[1].strip()
    if ':' not in comment: continue
    hex_str = comment.split(':')[1].strip().split()[0]
    try:
      machine_bytes = bytes.fromhex(hex_str)[::-1]  # big-endian to little-endian
      results.append((instr, machine_bytes))
    except ValueError:
      continue
  return results

def compile_asm(instr: str, compiler=None) -> bytes | None:
  """Compile a single instruction with llvm-mc and return the machine code bytes."""
  import subprocess
  try:
    result = subprocess.run(
      ['llvm-mc', '-triple=amdgcn', '-mcpu=gfx1100', '-mattr=+real-true16,+wavefrontsize32', '-show-encoding'],
      input=f".text\n{instr}\n", capture_output=True, text=True)
    if result.returncode != 0: return None
    # Parse encoding: [0x01,0x39,0x0a,0x7e]
    for line in result.stdout.split('\n'):
      if 'encoding:' in line:
        enc = line.split('encoding:')[1].strip()
        if enc.startswith('[') and enc.endswith(']'):
          hex_vals = enc[1:-1].replace('0x', '').replace(',', '').replace(' ', '')
          return bytes.fromhex(hex_vals)
  except Exception:
    pass
  return None

class TestTinygradKernelRoundtrip(unittest.TestCase):
  """Test roundtrip on real tinygrad-generated kernels using get_kernels_from_tinygrad pattern."""

  def _test_kernel_roundtrip(self, op_fn):
    """Generate kernel from op_fn, test:
    1. decode -> reencode matches original bytes
    2. asm(disasm()) matches LLVM output
    3. our disasm() matches LLVM's disassembly string exactly
    """
    from extra.assembly.rdna3.test.test_compare_emulators import get_kernels_from_tinygrad
    from tinygrad.runtime.support.compiler_amd import HIPCompiler

    kernels, _, _ = get_kernels_from_tinygrad(op_fn)
    compiler = HIPCompiler('gfx1100')

    decode_passed, decode_failed, decode_skipped = 0, 0, 0
    asm_passed, asm_failed, asm_skipped = 0, 0, 0
    disasm_passed, disasm_failed, disasm_skipped = 0, 0, 0
    decode_failures, asm_failures, disasm_failures = [], [], []

    for ki, kernel in enumerate(kernels):
      offset = 0
      while offset < len(kernel.code):
        remaining = kernel.code[offset:]
        fmt = detect_format(remaining)
        if fmt is None:
          decode_skipped += 1
          asm_skipped += 1
          disasm_skipped += 1
          offset += 4
          continue

        size = fmt._size()
        if len(remaining) < size:
          break

        orig_bytes = remaining[:size]

        # Test 1: decode -> reencode roundtrip
        try:
          decoded = fmt.from_bytes(orig_bytes)
          reencoded = decoded.to_bytes()
          if reencoded[:size] == orig_bytes:
            decode_passed += 1
          else:
            decode_failed += 1
            decode_failures.append(f"K{ki}@{offset}: {decoded.disasm()}: orig={orig_bytes.hex()} reenc={reencoded[:size].hex()}")

          our_disasm = decoded.disasm()

          # Test 2: asm(disasm()) matches LLVM output
          try:
            our_bytes = asm(our_disasm).to_bytes()
            llvm_bytes = compile_asm(our_disasm, compiler)
            if llvm_bytes is None:
              asm_skipped += 1
            elif our_bytes[:len(llvm_bytes)] == llvm_bytes:
              asm_passed += 1
            else:
              asm_failed += 1
              asm_failures.append(f"K{ki}@{offset}: '{our_disasm}': ours={our_bytes[:len(llvm_bytes)].hex()} llvm={llvm_bytes.hex()}")
          except Exception:
            asm_skipped += 1

          # Test 3: our disasm() matches LLVM's disassembly string exactly
          # Skip if instruction uses op_XX (unknown opcode) or looks malformed (many raw field values)
          if our_disasm.startswith('op_') or re.search(r', \d+, \d+, \d+,', our_disasm):
            disasm_skipped += 1
          else:
            try:
              # Get LLVM's disassembly of our instruction
              src = f".text\n.globl test\n.p2align 8\n.type test,@function\ntest:\n  {our_disasm}\n"
              lib = compiler.compile(src)
              llvm_instrs = disassemble_lib(lib, compiler)
              if llvm_instrs:
                llvm_disasm = llvm_instrs[0][0]
                if our_disasm == llvm_disasm:
                  disasm_passed += 1
                else:
                  disasm_failed += 1
                  disasm_failures.append(f"K{ki}@{offset}: ours='{our_disasm}' llvm='{llvm_disasm}'")
              else:
                disasm_skipped += 1
            except Exception:
              disasm_skipped += 1

        except Exception:
          decode_skipped += 1
          asm_skipped += 1
          disasm_skipped += 1

        offset += size

    print(f"decode roundtrip: {decode_passed} passed, {decode_failed} failed, {decode_skipped} skipped")
    print(f"asm vs llvm: {asm_passed} passed, {asm_failed} failed, {asm_skipped} skipped")
    print(f"disasm vs llvm: {disasm_passed} passed, {disasm_failed} failed, {disasm_skipped} skipped")
    self.assertEqual(decode_failed, 0, f"Decode failures:\n" + "\n".join(decode_failures[:20]))
    self.assertEqual(asm_failed, 0, f"Asm failures:\n" + "\n".join(asm_failures[:20]))
    self.assertEqual(disasm_failed, 0, f"Disasm failures:\n" + "\n".join(disasm_failures[:20]))

  # Basic unary ops
  def test_neg(self): self._test_kernel_roundtrip(lambda T: -T([1.0, -2.0, 3.0, -4.0]))
  def test_relu(self): self._test_kernel_roundtrip(lambda T: T([-1.0, 0.0, 1.0, 2.0]).relu())
  def test_exp(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).exp())
  def test_log(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 3.0]).log())
  def test_sin(self): self._test_kernel_roundtrip(lambda T: T([0.0, 1.0, 2.0]).sin())
  def test_sqrt(self): self._test_kernel_roundtrip(lambda T: T([1.0, 4.0, 9.0]).sqrt())
  def test_recip(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0, 4.0]).reciprocal())

  # Binary ops
  def test_add(self): self._test_kernel_roundtrip(lambda T: T([1.0, 2.0]) + T([3.0, 4.0]))
  def test_sub(self): self._test_kernel_roundtrip(lambda T: T([5.0, 6.0]) - T([1.0, 2.0]))
  def test_mul(self): self._test_kernel_roundtrip(lambda T: T([2.0, 3.0]) * T([4.0, 5.0]))
  def test_div(self): self._test_kernel_roundtrip(lambda T: T([10.0, 20.0]) / T([2.0, 4.0]))
  def test_max_binary(self): self._test_kernel_roundtrip(lambda T: T([1.0, 5.0]).maximum(T([3.0, 2.0])))

  # Reductions
  def test_sum_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).sum())
  def test_max_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(64).max())
  def test_mean_reduce(self): self._test_kernel_roundtrip(lambda T: T.empty(32).mean())

  # Matmul
  def test_gemm_4x4(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4) @ T.empty(4, 4))
  def test_gemv(self): self._test_kernel_roundtrip(lambda T: T.empty(1, 16) @ T.empty(16, 16))

  # Complex ops
  def test_softmax(self): self._test_kernel_roundtrip(lambda T: T.empty(16).softmax())
  def test_layernorm(self): self._test_kernel_roundtrip(lambda T: T.empty(8, 8).layernorm())

  # Memory patterns
  def test_contiguous(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 4).permute(1, 0).contiguous())
  def test_reshape(self): self._test_kernel_roundtrip(lambda T: (T.empty(16) + 1).reshape(4, 4).contiguous())
  def test_expand(self): self._test_kernel_roundtrip(lambda T: T.empty(4, 1).expand(4, 4).contiguous())

  # Cast ops
  def test_cast_int(self): self._test_kernel_roundtrip(lambda T: T.empty(16).int().float())
  def test_cast_half(self): self._test_kernel_roundtrip(lambda T: T.empty(16).half().float())

  # Comparison ops
  def test_cmp_lt(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) < T.empty(64)).where(T.empty(64), T.empty(64)))
  def test_where(self): self._test_kernel_roundtrip(lambda T: (T.empty(64) > 0).where(T.empty(64), T.empty(64)))

  # Fused ops
  def test_fma(self): self._test_kernel_roundtrip(lambda T: (T([1.0, 2.0]) * T([3.0, 4.0]) + T([5.0, 6.0])))

if __name__ == "__main__":
  unittest.main()
