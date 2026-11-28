import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.uop import Ops
from tinygrad.dtype import AddrSpace

class TestCrossCompile(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler
    try: CPULLVMCompiler(["x86_64", "x86-64"])
    except Exception as e: raise unittest.SkipTest(f"LLVM not available: {e}")

  def test_gpu_we_dont_have(self):
    # the compiler only gets the targets, a gfx1100 with an x86_64 host. it opens no device
    out = ((Tensor.empty(4, 4, device="AMD") @ Tensor.empty(4, 4, device="AMD")).to("CPU") + 1).contiguous()
    host, opened = "CPU:LLVM:x86_64,x86-64", set(Device._opened_devices)
    with Context(ALLOW_DEVICE_USAGE=0, DEV=f"AMD:LLVM:gfx1100;{host}", HCQ_RUNTIME_DEV=host): linear = Tensor.compile(out.export()[0])
    self.assertEqual(set(Device._opened_devices), opened)
    nodes = linear.toposort(enter_calls=True)
    self.assertFalse(any(u.op is Ops.BUFFER and u.addrspace is AddrSpace.GLOBAL for u in nodes))
    self.assertTrue(all(u.src[-1].op is Ops.BINARY for u in nodes if u.op is Ops.PROGRAM))

if __name__ == "__main__":
  unittest.main()
