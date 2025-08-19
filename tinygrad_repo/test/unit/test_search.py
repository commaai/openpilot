import unittest
from tinygrad import Tensor, Device
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.device import Buffer
from tinygrad.codegen.opt.search import get_test_global_size, bufs_from_lin
from tinygrad.helpers import GlobalCounters
from extra.optimization.helpers import time_linearizer
from test.test_linearizer import push_views

class TestSearchUtil(unittest.TestCase):
  def test_get_test_global_size(self):
    self.assertEqual(get_test_global_size([256, 256, 256], 65536, {}), ([256, 16, 16], 256.0))
    self.assertEqual(get_test_global_size([65536, 1, 1], 256, {}), ([256, 1, 1], 256.0))
    self.assertEqual(get_test_global_size([77, 1, 1], 16, {}), ([9, 1, 1], 77/9))

  def test_bufs_from_lin(self):
    a = Tensor([1,2,3,4]).realize()
    si = (a+1).schedule()[0]
    rawbufs = bufs_from_lin(Kernel(si.ast))
    assert len(rawbufs) == 2
    assert all(r is not None for r in rawbufs)
    assert all(isinstance(r, Buffer) for r in rawbufs)
    assert all(r.size > 0 for r in rawbufs)

  def test_bufs_from_lin_alt(self):
    a = Tensor.randn(4, 4).realize()
    b = a+a[0]
    si = b.schedule()[0]
    rawbufs = bufs_from_lin(Kernel(push_views(si.ast)))
    assert len(rawbufs) == 2
    assert all(r is not None for r in rawbufs)
    assert all(isinstance(r, Buffer) for r in rawbufs)
    assert all(r.size > 0 for r in rawbufs)

class TestTimeLinearizer(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WebGPU timestamps are low precision, tm is 0")
  def test_reasonable_time(self):
    a = Tensor([1,2,3,4]).realize()
    si = (a+1).schedule()[0]
    # create fresh empty buffers
    rawbufs = [Buffer(b.device, b.size, b.dtype).allocate() for b in si.bufs]
    tm = time_linearizer(Kernel(push_views(si.ast)), rawbufs, allow_test_size=False, cnt=10, disable_cache=True)
    assert tm > 0 and tm != float('inf')

  # Ensure that the kernel count is not incremented by time_linearizer when clearing l2
  def test_kernel_count(self):
    ast = Tensor.zeros(16).contiguous().kernelize().uop.src[1].arg.ast
    lin = Kernel(push_views(ast))
    bufs = bufs_from_lin(lin)

    kernel_count = GlobalCounters.kernel_count
    time_linearizer(lin, bufs, allow_test_size=False, cnt=2, disable_cache=True, clear_l2=True)
    assert GlobalCounters.kernel_count == kernel_count, "kernel count was incremented by time_linearizer"

if __name__ == "__main__":
  unittest.main()