import time, unittest
from tinygrad.runtime.support.hip_comgr import compile_hip
from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.engine.schedule import create_schedule
from tinygrad.codegen.kernel import Kernel

class TestHIPCompileSpeed(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT != "HIP", "only run on HIP")
  def test_hip_compile(self):
    a, b = Tensor([1,2,3,4,5]), Tensor([1,2,3,4,5])
    out = a + b
    lin = Kernel(create_schedule([out.lazydata])[-1].ast[0])
    lin.linearize()

    reference = """
#include <hip/hip_common.h>
    typedef long unsigned int size_t;
    extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);
    extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
    extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);
    extern "C" __attribute__((global))void {name}(int* data0, const int* data1, const int* data2) {{
      int gidx0 = __ockl_get_group_id(0); /* 5 */
      int val0 = data1[gidx0];
      int val1 = data2[gidx0];
      data0[gidx0] = (val0+val1);
    }}
    """

    def time_compile(code):
      st = time.perf_counter()
      compile_hip(code)
      return (time.perf_counter() - st) * 1000

    tinygrad_tm = min([time_compile(Device[Device.DEFAULT].renderer.render(f"test{i}", lin.uops)) for i in range(10)])
    ref_tm = min([time_compile(reference.format(name=f"test{i}")) for i in range(10)])
    print(f"tinygrad {tinygrad_tm:6.2f} ms")
    print(f"reference {ref_tm:6.2f} ms")
    assert (tinygrad_tm - ref_tm) <= 10
