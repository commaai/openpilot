import unittest
import numpy as np
from tinygrad import Device
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

if __name__ == '__main__':
  unittest.main()
