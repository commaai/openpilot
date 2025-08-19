import functools
import time
import unittest

from tinygrad import Tensor, TinyJit, GlobalCounters, Device
from tinygrad.helpers import getenv, Context
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters
from tinygrad.engine.realize import run_schedule

from extra.models import resnet
from examples.mlperf.initializers import Conv2dHeNormal, Linear
from examples.hlb_cifar10 import UnsyncedBatchNorm

# benchmark memory or kernel count: DEFAULT_FLOAT=HALF python test/external/external_benchmark_resnet.py
# benchmark speed:                  BEAM=2 JITCNT=10 DEFAULT_FLOAT=HALF python test/external/external_benchmark_resnet.py
# benchmark only one layer:         BEAM=2 DEFAULT_FLOAT=HALF python test/external/external_benchmark_resnet.py BenchmarkResnetTrain.test_layer1_2
# inspect:                          DEBUG=2 BEAM=2 DEFAULT_FLOAT=HALF python test/external/external_benchmark_resnet.py
# inspect 1x1 convs:                DEBUG=2 BEAM=2 CONV=2 DEFAULT_FLOAT=HALF python test/external/external_benchmark_resnet.py
# inspect 3x3 convs:                DEBUG=2 BEAM=2 CONV=2 DEFAULT_FLOAT=HALF python test/external/external_benchmark_resnet.py
# inspect 3x3 convs with batchnorm: DEBUG=2 BEAM=2 CONV=2 BN=1 DEFAULT_FLOAT=HALF python test/external/external_benchmark_resnet.py
# etc

# use ASSIGN=0 to disable batchnorm/optimizer assigns

# memory will be slightly high with JITCNT > 1

bs = getenv("BS", 64)

class BenchmarkResnetTrain(unittest.TestCase):
  def _get_layer(self, layer_i, slice_i):
    # isolate to conv, with or without BN
    conv = getenv("CONV", 0)
    bn = getenv("BN", 0)

    if not hasattr(self, 'model'):
      resnet.Conv2d = Conv2dHeNormal
      resnet.Linear = Linear
      if not getenv("SYNCBN"): resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=1)
      self.model = resnet.ResNet50()
      self.layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]

    layer = self.layers[layer_i][slice_i]
    xy = 112 >> layer_i
    xy >>= (1 if slice_i > 0 or layer_i == 0 else 0)  # layer 1 is preceded by maxpool2d
    name = f"layer{layer_i+1} slice{slice_i+1}"

    # get specific conv
    if conv:
      convs = [layer.conv1, layer.conv2, layer.conv3] + ([layer.downsample[0]] if layer.downsample else [])
      bns = [layer.bn1, layer.bn2, layer.bn3] + ([layer.downsample[1]] if layer.downsample else [])
      f = [convs[conv-1]]
      if bn: f.append(bns[conv-1])
      f.append(Tensor.relu)
      cin = f[0].in_channels
      if conv == 3: xy //= convs[1].stride
      return f"{name} conv{conv} x{str((bs, cin, xy, xy)):20s} k{str(f[0].weight.shape):20s}" + (" bn" if bn else ""), f, cin, xy

    cin = layer.conv1.in_channels
    return f"{name} x{(bs, cin, xy, xy)}", [layer], cin, xy
  def _test_layer(self, name, layer, cin, xy):
    optim = SGD(get_parameters(layer), bs / 128 * 1.0)  # need sgd for some params but not consequential for benchmarking
    with Context(TRACK_MATCH_STATS=0): Tensor.realize(*[t.assign(t.detach().contiguous()) for t in get_parameters(optim)])

    JITCNT = getenv("JITCNT", 1)
    Tensor.training = True
    @TinyJit
    def step(x):
      optim.zero_grad()
      x.grad = None

      y = x.sequential(layer).contiguous().contiguous_backward()
      y.sum().backward()
      if getenv("ASSIGN", 1): sched, _ = Tensor.schedule_with_vars(y, x.grad, *optim.schedule_step())
      else: sched, _ = Tensor.schedule_with_vars(y, x.grad, *[t.grad for t in optim.params])

      for _ in range(JITCNT):
        run_schedule(list(sched))

    CNT = getenv("CNT", 5)
    best_tm = None
    flops, mem_used, mem, kernels = None, None, None, None
    for i in range(CNT):
      with Context(TRACK_MATCH_STATS=0): x = Tensor.randn(bs, cin, xy, xy, requires_grad=True).realize()
      GlobalCounters.reset()

      st = time.perf_counter()
      step(x)
      Device[Device.DEFAULT].synchronize()
      et = time.perf_counter()

      flops = GlobalCounters.global_ops / JITCNT
      mem_used = GlobalCounters.mem_used  # a little high with JITCNT > 1 fsr
      mem = GlobalCounters.global_mem / JITCNT
      if kernels is None: kernels = GlobalCounters.kernel_count // JITCNT
      tm = (et-st) / JITCNT
      if best_tm is None or tm < best_tm: best_tm = tm
    print(f"\r{name:38s}: {best_tm * 1000:>9.2f} ms, {flops / 10**12 / best_tm:>6.2f} tflops, {mem / 10**9 / best_tm:>5.0f} GB/s, "
          f"{mem_used / 10**9: 6.2f} GB used, {kernels:>5d} kernels")
    return best_tm, flops, mem, kernels

  def test_layer1_1(self): self._est(*self._test_layer(*self._get_layer(0, 0)), 1)
  def test_layer1_2(self): self._est(*self._test_layer(*self._get_layer(0, 1)), 2)
  def test_layer2_1(self): self._est(*self._test_layer(*self._get_layer(1, 0)), 1)
  def test_layer2_2(self): self._est(*self._test_layer(*self._get_layer(1, 1)), 3)
  def test_layer3_1(self): self._est(*self._test_layer(*self._get_layer(2, 0)), 1)
  def test_layer3_2(self): self._est(*self._test_layer(*self._get_layer(2, 1)), 5)
  def test_layer4_1(self): self._est(*self._test_layer(*self._get_layer(3, 0)), 1)
  def test_layer4_2(self): self._est(*self._test_layer(*self._get_layer(3, 1)), 2)

  est_tm, est_flops, est_mem, est_kernels = 0, 0, 0, 0

  @classmethod
  def _est(cls, tm, flops, mem, kernels, mult):
    cls.est_tm += tm * mult
    cls.est_flops += flops * mult
    cls.est_mem += mem * mult
    cls.est_kernels += kernels * mult

  @classmethod
  def tearDownClass(cls):
    print(f"\restimated step tm: {cls.est_tm * 1000.0:.2f} ms, {cls.est_flops / 10 ** 12 / cls.est_tm:.3f} tflops, "
          f"{cls.est_mem / 10 ** 9 / cls.est_tm:.2f} GB/s, {cls.est_kernels} kernels")


if __name__ == '__main__':
  unittest.main()
