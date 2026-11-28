import gc, unittest
from unittest.mock import patch
from tinygrad import Tensor
from tinygrad.device import Device, Buffer, BufferSpec
from tinygrad.renderer import CompileError
from tinygrad.helpers import Context
if Device.DEFAULT=="METAL":
  from tinygrad.runtime.ops_metal import MetalAllocator, MetalCompiler
@unittest.skipIf(Device.DEFAULT!="METAL", "Metal support required")
class TestMetal(unittest.TestCase):
  def test_alloc_oom(self):
    device = Device["METAL"]
    with self.assertRaises(MemoryError):
      device.allocator.alloc(10000000000000000000)

  def test_compile_error(self):
    compiler = MetalCompiler()
    with self.assertRaises(CompileError):
      compiler.compile("this is not valid metal")

  def test_compile_success(self):
    compiler = MetalCompiler()
    ret = compiler.compile("""
#include <metal_stdlib>
  using namespace metal;
  kernel void E_4n1(device int* data0, const device int* data1, const device int* data2,
          uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    int val0 = *(data1+0);
    int val1 = *(data1+1);
    int val2 = *(data1+2);
    int val3 = *(data1+3);
    int val4 = *(data2+0);
    int val5 = *(data2+1);
    int val6 = *(data2+2);
    int val7 = *(data2+3);
    *(data0+0) = (val0+val4);
    *(data0+1) = (val1+val5);
    *(data0+2) = (val2+val6);
    *(data0+3) = (val3+val7);
  }
""")
    assert ret is not None

  def test_failed_newLibraryWithData(self):
    device = Device["METAL"]
    compiler = MetalCompiler()
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void r_5(device int* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]){
  data0[0] = 0;
}
""")
    with self.assertRaises(RuntimeError):
      compiled = compiled[:40] # corrupt the compiled program
      device.pipeline(compiled, "r_5")

  def test_cyclic_gc_keeps_cached_storage(self):
    device = Device['METAL']
    device.synchronize()
    gc.collect()
    allocator = MetalAllocator(device)
    with Context(LRU=1):
      # Like an LRU hit, storage predates its new Buffer and can be finalized first by cyclic GC.
      storage = allocator.alloc(4096)
      with patch.object(allocator, 'alloc', return_value=storage):
        buf = Buffer('METAL', 4096, allocator=allocator, preallocate=True)
      del storage
      holder = [buf]
      holder.append(holder)
      try:
        # Intercept release so a failing test doesn't leave a freed native resource in the allocator cache.
        with patch.object(type(buf.meta), 'release', autospec=True) as release:
          del buf, holder
          gc.collect()
          release.assert_not_called()
        self.assertEqual(len(allocator.cache[(4096, BufferSpec())]), 1)
      finally:
        allocator.free_cache()

  def test_multi_device_profile_synchronize(self):
    # the autorelease pool is shared by every METAL device, synchronizing one must not free the command buffers the other still profiles
    with Context(PROFILE=1):
      for _ in range(3):
        Tensor.ones(4, device="METAL:0").contiguous().realize()
        Tensor.ones(4, device="METAL:1").contiguous().realize()
        Device["METAL:1"].synchronize()
        Device["METAL:0"].synchronize()

  def test_free(self):
    size = 2**16
    device = Device['METAL']
    device.synchronize() # the device's timeline buffer is allocated on first use
    before = device.sysdevice.currentAllocatedSize()

    buf = device.allocator.alloc(size, BufferSpec(nolru=True))
    self.assertEqual(curr:=device.sysdevice.currentAllocatedSize(), before+size, msg=f"{curr=} - {before=}")
    device.allocator.free(buf, size, BufferSpec(nolru=True))
    self.assertEqual(curr:=device.sysdevice.currentAllocatedSize(), before, msg=f"{curr=} - {before=}")
