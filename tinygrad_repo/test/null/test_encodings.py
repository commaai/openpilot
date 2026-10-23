import unittest
from dataclasses import replace
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa.x86 import X86Ops, X86Renderer, RBP, RDI, RSP, RSI, RAX, RDX, XMM, GPR, imm, def_reg

def ins(op, dt, src, tag=None): return UOp(Ops.INS, arg=(op, dt), src=src, tag=tag)
# the operand width is carried by the register in the tag, not by the dtype
def reg(r:Register, size:int) -> Register: return replace(r, size=size)
# the element size of a memory operand comes from the dtype of its base
def ptr(r:Register, dt:DType) -> UOp: return UOp(Ops.INS, arg=(X86Ops.DEFINE, dt), tag=(r,))

class TestEncodingsX86(unittest.TestCase):
  # NOTE: x86 supports a single displacement as memory address and index without base memory address
  # these have no use cases so they aren't supported
  def encode(self, u:UOp): return X86Renderer(Target(device="CPU", arch="x86_64")).render([u])

  # displacement of 0 isn't emitted
  def test_base_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RDI, dtypes.int32), UOp(Ops.NOOP), imm(dtypes.int8, 0)), reg(RDI, 4))
    # mov edi, dword ptr [rdi]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 3F"))

  # rsp/r12 require a sib byte when used as base memory address
  def test_rsp_base_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RSP, dtypes.int32), UOp(Ops.NOOP), imm(dtypes.int8, 0)), reg(RSP, 4))
    # mov esp, dword ptr [rsp]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 24 24"))

  # rbp/r13 require a displacement when used as base memory address
  def test_rbp_base_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RBP, dtypes.int32), UOp(Ops.NOOP), imm(dtypes.int8, 0)), reg(RBP, 4))
    # mov ebp, dword ptr [rbp + 0]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 6D 00"))

  # test [base + index*scale]
  def test_base_index_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RAX, dtypes.int32), def_reg(RDX), imm(dtypes.int8, 0)), reg(RAX, 4))
    # mov eax, dword ptr [rax + rdx*4]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 04 90"))

  # rsp as index means no index
  def test_rsp_index_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RAX, dtypes.int32), def_reg(RSP), imm(dtypes.int8, 0)), reg(RAX, 4))
    # mov eax, dword ptr [rax]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 00"))

  # however r12 is a valid index
  def test_r12_index_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RAX, dtypes.int32), def_reg(GPR[12]), imm(dtypes.int8, 0)), reg(RAX, 4))
    # mov eax, dword ptr [rax + r12*4]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("42 8B 04 A0"))

  # test [base + index*scale + 8bit disp]
  def test_complex_address_8bit_disp(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RDI, dtypes.int32), def_reg(RSI), imm(dtypes.int8, 10)), reg(RDI, 4))
    # mov edi, dword ptr [rdi + rsi*4 + 0xa]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 7C B7 0A"))

  # test [base + index*scale + 32bit disp]
  def test_complex_address_32bit_disp(self):
    load = ins(X86Ops.MOV, dtypes.int32, (ptr(RDI, dtypes.int32), def_reg(RSI), imm(dtypes.int32, 10000)), reg(RDI, 4))
    # mov edi, dword ptr [rdi + rsi*4 + 0x2710]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B BC B7 10 27 00 00"))

  # 8bit variants of legacy instructions subtract 1 from opcode
  def test_8bit_legacy_encoding(self):
    cast = ins(X86Ops.MOVSX, dtypes.int32, (def_reg(reg(RDX, 1)),), reg(RAX, 4))
    # movsx eax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("0F BE C2"))

  # accessing lower 8 bits of rsp, rbp, rsi, rdi requires rex prefix
  def test_lower_8bits_reg(self):
    cast = ins(X86Ops.MOVSX, dtypes.int32, (def_reg(reg(RDI, 1)),), reg(RAX, 4))
    # movsx eax, dil
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("40 0F BE C7"))

  # test 16 bit variant of legacy instruction
  def test_16bit_legacy_encoding(self):
    cast = ins(X86Ops.MOVSX, dtypes.int16, (def_reg(reg(RDX, 1)),), reg(RAX, 2))
    # movsx ax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("66 0F BE C2"))

  # test 64 bit variant of legacy instruction
  def test_64bit_legacy_encoding(self):
    cast = ins(X86Ops.MOVSX, dtypes.int64, (def_reg(reg(RDX, 1)),), reg(RAX, 8))
    # movsx rax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("48 0F BE C2"))

  # the width comes from the register in the tag, the dtype doesn't take part in the encoding
  def test_width_from_register(self):
    for size, expected in ((1, "8A C2"), (2, "66 8B C2"), (4, "8B C2"), (8, "48 8B C2")):
      mov = ins(X86Ops.MOV, dtypes.float32, (def_reg(reg(RDX, size)),), reg(RAX, size))
      self.assertEqual(bytes.fromhex(self.encode(mov)), bytes.fromhex(expected))

  # test compact vex encoding
  def test_compact_vex_encoding(self):
    xmm0, xmm1 = def_reg(XMM[0]), def_reg(XMM[1])
    add = ins(X86Ops.VADDSS, dtypes.float32, (xmm0, xmm1), XMM[0])
    # vaddss xmm0, xmm0, xmm1
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C5 FA 58 C1"))

  # test long vex encoding
  def test_long_vex_encoding(self):
    xmm0, xmm8 = def_reg(XMM[0]), def_reg(XMM[8])
    add = ins(X86Ops.VADDSS, dtypes.float32, (xmm0, xmm8), XMM[0])
    # vaddss xmm0, xmm0, xmm8
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C4 C1 7A 58 C0"))

  # test encoding where register is in the immediate field
  def test_reg_in_imm_field(self):
    xmm0, xmm1, xmm2 = def_reg(XMM[0]), def_reg(XMM[1]), def_reg(XMM[2])
    blend = ins(X86Ops.VBLENDVPS, dtypes.float32, (xmm0, xmm1, xmm2), XMM[0])
    # vblendvps xmm0, xmm0, xmm1, xmm2
    self.assertEqual(bytes.fromhex(self.encode(blend)), bytes.fromhex("C4 E3 79 4A C1 20"))

  # when writting to mem the uop takes the store form where dtype is void and there's no definition
  def test_write_mem(self):
    address = (ptr(RDI, dtypes.int32), def_reg(RSI), imm(dtypes.int8, 10))
    xmm0 = def_reg(XMM[0])
    extr = ins(X86Ops.VPEXTRD, dtypes.void, address + (xmm0, imm(dtypes.uint8, 0)))
    # vpextrd dword ptr [rdi + rsi*4 + 0xa], xmm0, 0
    self.assertEqual(bytes.fromhex(self.encode(extr)), bytes.fromhex("C4 E3 79 16 44 B7 0A 00"))

  # test two address instruction with fused load works
  def test_two_address_load(self):
    address = (ptr(RDI, dtypes.int32), def_reg(RSI), imm(dtypes.int8, 10))
    cmove = ins(X86Ops.CMOVE, dtypes.int32, address, reg(RAX, 4))
    # cmove eax, dword ptr [rdi + rsi*4 + 0xa]
    self.assertEqual(bytes.fromhex(self.encode(cmove)), bytes.fromhex("0F 44 44 B7 0A"))

  # test instruction where displacement and imm have the same value
  def test_disp_imm_same_value(self):
    address = (ptr(RDI, dtypes.int8), def_reg(RSI), imm(dtypes.int8, 10))
    mov = ins(X86Ops.MOVi, dtypes.void, address + (imm(dtypes.int8, 10),))
    # mov byte ptr [rdi + rsi + 0xa], 0xa
    self.assertEqual(bytes.fromhex(self.encode(mov)), bytes.fromhex("40 C6 44 37 0A 0A"))

    address = (ptr(RDI, dtypes.int32), def_reg(RSI), imm(dtypes.int32, 10))
    imul = ins(X86Ops.IMULi, dtypes.int32, address + (imm(dtypes.int32, 10),), reg(RDI, 4))
    # imul edi, dword ptr [rdi + rsi*4 + 0xa], 0xa
    self.assertEqual(bytes.fromhex(self.encode(imul)), bytes.fromhex("69 BC B7 0A 00 00 00 0A 00 00 00"))

  # cmoves have the cmp as the last src even though it is not explicitly used, the cmp doesn't define a reg and is ignored in the encoding
  def test_cmove_ignore_cmp(self):
    cmove = ins(X86Ops.CMOVE, dtypes.int32, (def_reg(reg(RAX, 4)), UOp(Ops.INS, arg=(X86Ops.CMP, dtypes.void))), reg(RDX, 4))
    # cmove edx, eax
    self.assertEqual(bytes.fromhex(self.encode(cmove)), bytes.fromhex("0F 44 D0"))

if __name__ == "__main__":
  unittest.main()
