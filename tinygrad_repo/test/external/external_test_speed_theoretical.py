import time, unittest
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import getenv, GlobalCounters

SZMAX = getenv("SZMAX", 10)
SZMIN = min(SZMAX, getenv("SZMIN", 10))
def _test(tcount, fxn, dtype=dtypes.float):
  print(f"**** testing {fxn.__name__} {dtype}")
  allgbs = []
  for sz in range(SZMIN, SZMAX+1):
    jfxn = TinyJit(fxn)
    ts = [Tensor.zeros((2**sz)*1024*1024, dtype=dtype).contiguous().realize() for _ in range(tcount)]
    tms = []
    for _ in range(10):
      ts = [(x+1).realize() for x in ts]
      Device.default.synchronize()
      GlobalCounters.global_ops = 0
      GlobalCounters.global_mem = 0
      st = time.perf_counter()
      jfxn(*ts).nbytes()
      Device.default.synchronize()
      tms.append(time.perf_counter() - st)
      ops, mem = GlobalCounters.global_ops, GlobalCounters.global_mem
    gflops = ops*1e-9/min(tms)
    gbs = mem*1e-9/min(tms)
    print(f"{ts[0].nbytes()/(1024*1024):10.0f} MB, {min(tms)*1e3:6.2f} ms {gbs:10.2f} GB/s {gflops:10.2f} GFLOPS {str(ts[0].shape):20s}")
    allgbs.append(gbs)
  return max(allgbs)

MEMBW = getenv("MEMBW", 10)
class TestRamBandwidth(unittest.TestCase):
  def test_add(self): self.assertGreater(_test(2, Tensor.add), MEMBW)
  def test_exp(self): self.assertGreater(_test(1, Tensor.exp), MEMBW)
  def test_sum(self): self.assertGreater(_test(1, Tensor.sum), MEMBW)

# ratio between MEM and FLOPS < 1000
# NOTE: On AMD, (x*x)+1 gets ~30 TFLOPS, (x*x)+3 gets ~60 TFLOPS
def flopsmax(x):
  for _ in range(500): x = (x*x)+3
  return x

class TestFlops(unittest.TestCase):
  def test_flops_int8(self): _test(1, flopsmax, dtypes.int8)
  def test_flops_fp16(self): _test(1, flopsmax, dtypes.half)
  def test_flops_fp32(self): _test(1, flopsmax)

if __name__ == '__main__':
  unittest.main()
