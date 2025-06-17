import unittest
import numpy as np
from tinygrad import Device
from tinygrad.device import CompileError
from tinygrad.helpers import flat_mv
if Device.DEFAULT=="AMD":
  from tinygrad.runtime.ops_amd import AMDAllocator, AMDDevice, AMDProgram
  from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDLLVM(unittest.TestCase):
  def test_compiler(self):
    src = '''
; https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/imm.ll
define amdgpu_kernel void @i64_imm_inline_lo(ptr addrspace(1) %out) {
entry:
  store i64 1311768464867721221, ptr addrspace(1) %out ; 0x1234567800000005
  ret void
}
    '''
    device = AMDDevice()
    compiler = AMDLLVMCompiler("gfx1100")
    obj = compiler.compile(src)
    allocator = AMDAllocator(device)
    a = allocator.alloc(1*8)
    prog = AMDProgram(device, "test", obj)
    prog(a, wait=True)
    na = np.empty(1, np.uint64)
    allocator._copyout(flat_mv(na.data), a)
    assert na == [0x1234567800000005]

  def test_compiler_diag_error(self):
    src = """
@local_temp0 = internal unnamed_addr addrspace(3) global [{N} x float*] undef, align 16
define amdgpu_kernel void @test(float* noalias align 32 %data0, half* noalias align 32 %data1, float* noalias align 32 %data2) #0
{{
  %local_temp0 = addrspacecast [{N} x float*] addrspace(3)* @local_temp0 to [{N} x float*]*
  %v178 = getelementptr inbounds float, float* %local_temp0, i32 1
  %v133 = getelementptr inbounds float, float* %data2, i32 1
  %v134 = load float, float* %v133
  store float %v134, float* %v178
  ret void
}}
"""
    compiler = AMDLLVMCompiler("gfx1100")
    compiler.compile(src.format(N=65536//8))
    with self.assertRaises(CompileError):
      # llvm diagnostic: <unknown>:0:0: local memory (65544) exceeds limit (65536) in function 'test'
      compiler.compile(src.format(N=65536//8+1))

if __name__ == '__main__':
  unittest.main()
