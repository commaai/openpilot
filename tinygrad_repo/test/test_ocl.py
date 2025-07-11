import unittest
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.helpers import CI
from tinygrad.runtime.ops_gpu import CLDevice, CLAllocator, CLCompiler, CLProgram

@unittest.skipUnless(Device.DEFAULT == "GPU", "Runs only on OpenCL (GPU)")
class TestCLError(unittest.TestCase):
  @unittest.skipIf(CI, "dangerous for CI, it allocates tons of memory")
  def test_oom(self):
    with self.assertRaises(RuntimeError) as err:
      allocator = CLAllocator(CLDevice())
      for i in range(1_000_000):
        allocator.alloc(1_000_000_000)
    assert str(err.exception) == "OpenCL Error -6: CL_OUT_OF_HOST_MEMORY"

  def test_invalid_kernel_name(self):
    device = Device[Device.DEFAULT]
    with self.assertRaises(RuntimeError) as err:
      CLProgram(device, name="", lib=CLCompiler(device, "test").compile("__kernel void test(__global int* a) { a[0] = 1; }"))
    assert str(err.exception) == "OpenCL Error -46: CL_INVALID_KERNEL_NAME"

  def test_unaligned_copy(self):
    data = list(range(65))
    unaligned = memoryview(bytearray(data))[1:]
    buffer = Buffer("GPU", 64, dtypes.uint8).allocate()
    buffer.copyin(unaligned)
    result = memoryview(bytearray(len(data) - 1))
    buffer.copyout(result)
    assert unaligned == result, "Unaligned data copied in must be equal to data copied out."
