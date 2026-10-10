import gc, unittest
from tinygrad import Tensor, Device, nn, GlobalCounters, TinyJit, dtypes, UOp
from tinygrad.uop.ops import Ops
from tinygrad.helpers import Context
from tinygrad.nn.state import get_parameters, get_state_dict
from test.helpers import not_support_multi_device, needs_second_gpu

d1 = f"{Device.DEFAULT}:1"
d2 = f"{Device.DEFAULT}:2"
d3 = f"{Device.DEFAULT}:3"
d4 = f"{Device.DEFAULT}:4"
devices_2 = (d1, d2)
devices_3 = (d1, d2, d3)
devices_4 = (d1, d2, d3, d4)

class TestMultiRamUsage(unittest.TestCase):
  def setUp(self):
    self.enterContext(Context(DEV="NULL"))
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
    self.assertUsed(0)

  def test_sharded_memory_axis_const(self):
    devices_4 = tuple(f"NULL:{i+1}" for i in range(4))
    X = Tensor.ones(256, buffer=False).realize()
    self.assertUsed(0)
    X.shard_(devices_4, axis=0).realize()
    self.assertUsed(0)

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
      inner = Tensor(t.uop.src[0]) if t.uop.op is Ops.UNSHARD else t
      return (inner.sum(),)
    param = x.as_param(0)
    fxn = _fxn(param.uop, x.device)
    per_dev_scalar = Tensor(fxn[0].uop.call_with_output(x.uop))
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

  def test_uop_shard_axis_none(self):
    devices = ("NULL:0", "NULL:1")
    u = Tensor.ones(8).contiguous().realize().uop
    self.assertIsNone(u.shard(devices).axis)
    self.assertEqual(u.shard(devices, 0).axis, 0)

  def test_empty_like_sharded(self):
    t = Tensor.ones(4, 8).shard(("NULL:0", "NULL:1"), axis=0)
    e = t.empty_like()
    self.assertEqual(e.shape, t.shape)
    self.assertEqual(e.device, t.device)
    self.assertEqual(e.uop.axis, 0)
    self.assertTrue(e.uop.has_buffer_identity())

  def test_symbolic_reshape_shard_axis(self):
    rows = UOp.variable("rows", 1, 4).bind(3)
    x = Tensor.empty(4, 2).shard(("NULL:1", "NULL:2"), axis=1)[:rows]
    self.assertEqual(x.reshape(rows, 1, 2).uop.axis, 2)

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestMultiTensor(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  def test_shard_like(self):
    X = Tensor.ones(256).shard(devices_2, 0)
    Y = Tensor.zeros(256).shard_like(X)
    self.assertEqual(Y.device, X.device)
    self.assertEqual(Y.uop.axis, 0)
    # also test with axis=None
    X2 = Tensor.ones(256).shard(devices_2, axis=None)
    Y2 = Tensor.zeros(256).shard_like(X2)
    self.assertEqual(Y2.device, X2.device)
    self.assertEqual(Y2.uop.axis, None)
    # test with single device
    X3 = Tensor.ones(256)
    Y3 = Tensor.zeros(256).shard_like(X3)
    self.assertEqual(Y3.device, X3.device)
    # cannot shard_like multi unless it's a no-op
    X4 = Tensor.ones(256).shard(devices_2, 0)
    Y4 = Tensor.ones(256).shard(devices_2, 0).shard_like(X4)
    self.assertEqual(Y4.device, X4.device)
    self.assertEqual(Y4.uop.axis, 0)
    with self.assertRaises(RuntimeError):
      Tensor.ones(256).shard(devices_2, None).shard_like(X4)

  def test_shard_not_multiple(self):
    X = Tensor.ones(256).contiguous().realize()
    with self.assertRaises(RuntimeError):
      X.shard_(devices_3, 0)

  def test_shard_reshape_cross_boundary(self):
    X = Tensor.ones(5, 4).contiguous().realize().shard(devices_2, 1)
    with self.assertRaises(RuntimeError): X.reshape(10, 2).uop.axis

  def test_bn_ast_on_devices(self):
    t = Tensor.empty((16, 64, 112, 112)).shard(devices_4, axis=0)
    bn = nn.BatchNorm2d(64)
    for p in get_parameters(bn): p.shard_(devices_4).realize()

    out = bn(t)
    scheds = [call for call in out.schedule_linear().src if call.src[0].op is not Ops.STORE and set(call.device) <= set(devices_4)]
    self.assertEqual(set(scheds[0].device), set(devices_4), "should have ast on each shard device")
    self.assertEqual(len(set(s.src[0] for s in scheds)), 1)

  def test_init_rand_with_multiple_devices_fail(self):
    # init rand with multi device is not allowed
    with self.assertRaises(ValueError):
      Tensor.rand(256, device=devices_2)

  def test_rand_like_from_alu(self):
    a = Tensor.ones(4, 4).shard(devices_4, axis=0)
    aa = a + a
    self.assertEqual(aa.device, devices_4)
    self.assertEqual(aa.uop.axis, 0)
    raa = aa.rand_like()
    self.assertEqual(raa.device, devices_4)
    self.assertEqual(raa.uop.axis, 0)

    b = Tensor.empty(4, 4).shard(devices_4, axis=None)
    ab = a + b
    self.assertEqual(ab.device, devices_4)
    self.assertEqual(ab.uop.axis, 0)
    rab = ab.rand_like()
    self.assertEqual(rab.device, devices_4)
    self.assertEqual(rab.uop.axis, 0)

  def test_rand_like_none_shard(self):
    t = Tensor.empty((16, 16)).shard(devices_2)
    t2 = Tensor.rand_like(t)
    self.assertEqual(t.shape, t2.shape)
    self.assertEqual(t.device, t2.device)
    self.assertEqual(t.dtype, t2.dtype)
    self.assertEqual(t.uop.axis, t2.uop.axis)

  def test_rand_like_arg_dtype(self):
    t = Tensor.empty((16, 16), dtype=dtypes.int32).shard(devices_2, axis=1)
    t2 = Tensor.rand_like(t, dtype=dtypes.float32)
    self.assertEqual(t.dtype, dtypes.int32)
    self.assertEqual(t2.dtype, dtypes.float32)

  def test_rand_like_arg_device(self):
    # axis=None
    t = Tensor.empty((16, 16)).shard((d1, d2), axis=None)
    with self.assertRaises(RuntimeError):
      Tensor.rand_like(t, device=(d3, d4))

    # axis=1
    t = Tensor.empty((16, 16)).shard((d1, d2), axis=1)
    with self.assertRaises(RuntimeError):
      Tensor.rand_like(t, device=(d3, d4))

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestBatchNorm(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  def test_synced_vs_unsynced_bn(self):
    from examples.hlb_cifar10 import UnsyncedBatchNorm
    from tinygrad.nn import BatchNorm2d
    devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]
    x = Tensor.ones(8, 8, 8, 8).contiguous().realize().shard(devices, axis=0)

    with Context(TRAINING=1):
      synced_bn = BatchNorm2d(8)
      unsynced_bn = UnsyncedBatchNorm(8, num_devices=len(devices))

      for p in get_parameters(synced_bn):
        p.shard_(devices)
      for k, p in get_state_dict(unsynced_bn).items():
        if 'running_mean' in k or 'running_var' in k:
          p.shard_(devices, axis=0)
        else:
          p.to_(devices)

      synced_out = synced_bn(x)
      synced_si = list(synced_out.schedule_linear().src)
      unsynced_out = unsynced_bn(x)
      unsynced_si = list(unsynced_out.schedule_linear().src)

    # TODO: test synced / unsynced batchnorm cross device kernel and copies
    assert synced_si
    assert unsynced_si

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestBackendMultiTensor(unittest.TestCase):
  @needs_second_gpu
  def setUp(self): pass

  def test_shard_invalids_contiguous(self):
    # every store is Invalid, so none of them should become a (empty) kernel
    t = Tensor.invalids(8).shard(devices_2, axis=0).contiguous()
    self.assertEqual(len([c for c in t.schedule_linear().src if c.src[0].op is Ops.SINK]), 1)

if __name__ == '__main__':
  unittest.main()
