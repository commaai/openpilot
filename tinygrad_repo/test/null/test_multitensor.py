import gc, unittest
from tinygrad import Tensor, GlobalCounters, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context

class TestMultiRamUsage(unittest.TestCase):
  def setUp(self):
    gc.collect()
    self.baseline = GlobalCounters.mem_used
    self.baseline_per_device = dict(GlobalCounters.mem_used_per_device)
    self.N = 100
  def assertUsed(self, amt, strict=True):
    gc.collect()
    used = GlobalCounters.mem_used - self.baseline
    print(f"used {used} bytes")
    if strict: self.assertEqual(used, amt)
    else: self.assertLessEqual(used, amt)
  def assertDeviceUsed(self, expected:dict[str, int]):
    gc.collect()
    for dev, amt in expected.items():
      used = GlobalCounters.mem_used_per_device[dev] - self.baseline_per_device.get(dev, 0)
      self.assertEqual(used, amt, f"device {dev}: expected {amt} bytes used, got {used}")

  def test_zeros(self):
    _ = Tensor.zeros(self.N, self.N).contiguous().realize()
    self.assertUsed(self.N*self.N*4)

  def test_zeros_del(self):
    _ = Tensor.zeros(self.N, self.N).contiguous().realize()
    del _
    self.assertUsed(0)

  def test_zeros_copy(self):
    devices_2 = ("NULL:1", "NULL:2")
    _ = Tensor.zeros(self.N, self.N).contiguous().to(devices_2).realize()
    # NOTE: the first one on the DEFAULT device should be freed
    self.assertUsed(self.N*self.N*4*2)

  def test_zeros_shard(self, devices=("NULL:1", "NULL:2")):
    _ = Tensor.zeros(self.N, self.N).contiguous().shard(devices, axis=0).realize()
    self.assertUsed(self.N*self.N*4) # sharding should not increase total ram usage
  def test_zeros_shard_self(self): self.test_zeros_shard(("NULL:0", "NULL:1"))

  def test_zeros_contiguous_shard(self):
    devices_2 = ("NULL:1", "NULL:2")
    _ = Tensor.zeros(self.N, self.N).contiguous().shard(devices_2, axis=0).contiguous().realize()
    self.assertUsed(self.N*self.N*4) # sharding should not increase total ram usage

  def test_sharded_memory_replicated(self):
    devices_4 = tuple(f"NULL:{i+1}" for i in range(4))
    X = Tensor.ones(256).contiguous().realize()
    self.assertUsed(256 * 4)
    X.shard_(devices_4).realize()
    self.assertUsed(256 * 4 * 4)

  def test_sharded_memory_replicated_const(self):
    devices_4 = tuple(f"NULL:{i+1}" for i in range(4))
    X = Tensor.ones(256, buffer=False).realize()
    self.assertUsed(0)
    X.shard_(devices_4).realize()
    self.assertUsed(256 * 4 * 4)  # TODO: can be zero

  def test_sharded_memory_axis_const(self):
    devices_4 = tuple(f"NULL:{i+1}" for i in range(4))
    X = Tensor.ones(256, buffer=False).realize()
    self.assertUsed(0)
    X.shard_(devices_4, axis=0).realize()
    self.assertUsed(256 * 4)  # TODO: can be zero

  def test_zeros_per_device(self):
    _ = Tensor.zeros(self.N, self.N, device="NULL").contiguous().realize()
    self.assertDeviceUsed({"NULL": self.N*self.N*4})

  def test_zeros_del_per_device(self):
    _ = Tensor.zeros(self.N, self.N, device="NULL").contiguous().realize()
    del _
    self.assertDeviceUsed({"NULL": 0})

  def test_zeros_copy_per_device(self):
    devices_2 = ("NULL:1", "NULL:2")
    _ = Tensor.zeros(self.N, self.N).contiguous().to(devices_2).realize()
    self.assertDeviceUsed({"NULL:1": self.N*self.N*4, "NULL:2": self.N*self.N*4})

  def test_zeros_shard_per_device(self):
    devices_2 = ("NULL:1", "NULL:2")
    _ = Tensor.zeros(self.N, self.N).contiguous().shard(devices_2, axis=0).realize()
    self.assertDeviceUsed({"NULL:1": self.N*(self.N//2)*4, "NULL:2": self.N*(self.N//2)*4})

  def test_sharded_memory_replicated_per_device(self):
    devices_4 = tuple(f"NULL:{i+1}" for i in range(4))
    X = Tensor.ones(256, device="NULL").contiguous().realize()
    self.assertDeviceUsed({"NULL": 256*4})
    X.shard_(devices_4).realize()
    for d in devices_4:
      self.assertDeviceUsed({d: 256*4})

  def _test_matmul_half(self, dev_count:int):
    N = 32
    total_mem = {}
    devs = tuple(f"NULL:{i}" for i in range(dev_count))
    for dtype in {dtypes.float, dtypes.half}:
      GlobalCounters.reset()
      a = Tensor.empty((N, N), dtype=dtype, device=devs[0]).shard(devs, axis=0)
      b = Tensor.empty((N, N), dtype=dtype, device=devs[0]).shard(devs, axis=None)
      (a @ b).realize()
      total_mem[dtype] = GlobalCounters.global_mem
    self.assertEqual(total_mem[dtypes.half], total_mem[dtypes.float] // 2)

  def test_matmul_half(self): self._test_matmul_half(dev_count=2)
  def test_matmul_half_alt(self): self._test_matmul_half(dev_count=4)

  def test_multi_layer_allreduce(self):
    N = 32
    devices_2 = ("NULL:1", "NULL:2")

    def make_inp():
      x = Tensor.zeros(N, N).contiguous().shard(devices_2, axis=None).realize()
      w1 = Tensor.zeros(N, N).contiguous().shard(devices_2, axis=1).realize()
      w2 = Tensor.zeros(N, N).contiguous().shard(devices_2, axis=0).realize()
      return x, w1, w2

    def run_layers(n_layers):
      GlobalCounters.reset()

      @TinyJit
      def f(x, w1, w2):
        for _ in range(n_layers):
          x = (x @ w1 @ w2)
        return x.contiguous()

      for _ in range(3):
        a = make_inp()
        r = f(*a)
        del a, r

      gc.collect()
      return GlobalCounters.mem_used

    mem_2 = run_layers(2)
    mem_4 = run_layers(4)
    self.assertEqual(mem_2, mem_4, f"graph memory should not grow with layers: 2 layers={mem_2}, 4 layers={mem_4}")

  def test_allreduce_cast_dtype_memory(self):
    N = 32
    devices_2 = ("NULL:1", "NULL:2")
    mem = {}
    for allreduce_cast in (0, 1):
      GlobalCounters.reset()
      with Context(ALLREDUCE_CAST=allreduce_cast, SCACHE=0):
        x = Tensor.empty((N, N), dtype=dtypes.bfloat16, device="NULL:1").shard(devices_2, axis=0)
        x.sum(0).realize()
      mem[allreduce_cast] = GlobalCounters.global_mem
    # with ALLREDUCE_CAST, allreduce copies happen in bf16 (2 bytes) instead of fp32 (4 bytes)
    self.assertLess(mem[1], mem[0])

class TestMultiScalarALU(unittest.TestCase):
  """Test that tuple-device scalars work correctly in ALU with MULTI tensors (_shard scalar fix)."""
  def test_multi_times_replicated_scalar(self):
    devices = ("NULL:0", "NULL:1")
    x = Tensor.ones(4).contiguous().shard(devices, axis=0)
    s = Tensor(2.0).to(devices)
    result = x * s
    self.assertEqual(result.shape, (4,))
    self.assertEqual(result.uop.axis, 0)

  def test_multi_add_replicated_scalar(self):
    devices = ("NULL:0", "NULL:1")
    x = Tensor.ones(4).contiguous().shard(devices, axis=0)
    s = Tensor(1.0).to(devices)
    result = x + s
    self.assertEqual(result.shape, (4,))
    self.assertEqual(result.uop.axis, 0)

  def test_multi_times_call_scalar(self):
    """Per-device scalar from a CALL (like FP8 local amax) used in ALU with MULTI."""
    import functools
    from tinygrad.uop.ops import Ops
    devices = ("NULL:0", "NULL:1")
    x = Tensor.ones(4, 4).contiguous().shard(devices, axis=0)
    # simulate per-device scalar via CALL (strips MULTI from param body → no allreduce)
    @functools.cache
    def _fxn(x_p, device):
      t = Tensor(x_p, device=device)
      inner = Tensor(t.uop.src[0]) if t.uop.op is Ops.MULTI else t
      return (inner.sum(),)
    param = x.as_param(0)
    fxn = _fxn(param.uop, x.device)
    per_dev_scalar = Tensor(fxn[0].uop.call(x.uop).gettuple(0))
    result = x * per_dev_scalar
    self.assertEqual(result.shape, (4, 4))
    self.assertEqual(result.uop.axis, 0)
    result.realize()

class TestMultiAxis(unittest.TestCase):
  def test_reshape_shard_invalid(self):
    devices = ("NULL:0", "NULL:1")
    t = Tensor.ones(4, 3).shard(devices, axis=0)
    with self.assertRaises(RuntimeError, msg="reshape cannot move items between shards"):
      t.reshape(3, 4).uop.axis

  def test_reshape_shard_valid(self):
    devices = ("NULL:0", "NULL:1")
    t = Tensor.ones(4, 8).shard(devices, axis=0)
    self.assertEqual(t.reshape(2, 16).uop.axis, 0)
    self.assertEqual(t.reshape(2, 2, 8).uop.axis, 0)

  def test_empty_like_sharded(self):
    t = Tensor.ones(4, 8).shard(("NULL:0", "NULL:1"), axis=0)
    e = t.empty_like()
    self.assertEqual(e.shape, t.shape)
    self.assertEqual(e.device, t.device)
    self.assertEqual(e.uop.axis, 0)
    self.assertTrue(e.uop.has_buffer_identity())

if __name__ == '__main__':
  unittest.main()
