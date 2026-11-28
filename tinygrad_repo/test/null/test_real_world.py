import unittest, time, gc
import numpy as np
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit
from tinygrad import Tensor, Device, GlobalCounters, dtypes, Variable
from tinygrad.helpers import Context
from test.helpers import jit_cache_count, KernelCountException
from extra.lr_scheduler import OneCycleLR
from test.helpers import derandomize_model

from examples.gpt2 import Transformer as GPT2Transformer
from examples.hlb_cifar10 import SpeedyResNet, hyp
from extra.models.llama import Transformer as LLaMaTransformer
from extra.models.unet import ResBlock

global_mem_used = 0
def helper_test(nm, gen, model, max_memory_allowed, max_kernels_allowed, all_jitted=False):
  with Context(JIT=2):
    tms = []
    for _ in range(2):
      early_gen = [x.realize() if isinstance(x, Tensor) else x for x in gen()]
      GlobalCounters.reset()
      Device[Device.DEFAULT].synchronize()
      st = time.perf_counter_ns()
      model(*early_gen)
      Device[Device.DEFAULT].synchronize()
      tms.append(time.perf_counter_ns() - st)
    mem_used = (GlobalCounters.mem_used - global_mem_used) / 1e9

    kernels_used = jit_cache_count(model.captured._linear) if getattr(model, "captured", None) is not None else None
    print(f"{nm}: used {mem_used/1e9:.2f} GB and {kernels_used} kernels in {min(tms)/1e6:.2f} ms")
    assert mem_used < max_memory_allowed, f"{nm} used more than {max_memory_allowed:.3f} GB - {mem_used:.3} GB used"
    assert (max_memory_allowed - mem_used) / max_memory_allowed < 0.2, f"{max_memory_allowed:.3f} GB is too far from {mem_used:.3} GB used"
    if kernels_used:
      if kernels_used > max_kernels_allowed: raise KernelCountException(max_kernels_allowed, kernels_used)
      if (max_kernels_allowed - kernels_used) / max_kernels_allowed >= 0.2:
        raise KernelCountException(max_kernels_allowed, kernels_used)
    if all_jitted:
      assert kernels_used > 0 and kernels_used == GlobalCounters.kernel_count or (kernels_used <= GlobalCounters.kernel_count and getattr(Device[Device.DEFAULT], "graph", None)), f"only {kernels_used} out of {GlobalCounters.kernel_count} were jitted"  # noqa: E501

supported_dtypes = Device["NULL"].renderer.supported_dtypes()

class TestRealWorld(unittest.TestCase):
  def setUp(self):
    # kernel counts and memory are checked on NULL, a real device adds its runtime buffers (queues, programs, command buffers) to the memory used
    self.enterContext(Context(DEV="NULL"))
    gc.collect()
    global global_mem_used
    global_mem_used = GlobalCounters.mem_used
    np.random.seed(2002)

  def test_unet_resblock(self):
    model = [ResBlock(16, 24, 16) for _ in range(4)]
    derandomize_model(model)
    @TinyJit
    def test(t, t2):
      for l in model: t = l(t, t2)
      return t.realize()

    helper_test("test_unet_resblock", lambda: (Tensor.empty(4, 16, 8, 8), Tensor.empty(1, 24)), test, 0.0002, 37)

  @unittest.skipUnless(dtypes.float16 in supported_dtypes, "need dtypes.float16")
  def test_llama(self):
    self.enterContext(Context(DEFAULT_FLOAT=dtypes.float16))

    args_tiny = {"dim": 1024, "hidden_dim": 2048, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    model = LLaMaTransformer(**args_tiny)
    derandomize_model(model)
    @TinyJit
    def test(t): return model(t, 0).realize()
    # TODO: test first token vs rest properly
    helper_test("test_llama", lambda: (Tensor([[1,2,3,4]]),), test, 0.23, 118, all_jitted=True)

  @unittest.skipUnless(dtypes.float16 in supported_dtypes, "need dtypes.float16")
  def test_gpt2(self):
    self.enterContext(Context(DEFAULT_FLOAT=dtypes.float16))

    args_tiny = {"dim": 1024, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-5, "vocab_size": 1000}
    model = GPT2Transformer(**args_tiny)
    derandomize_model(model)
    @TinyJit
    def test(t, v):
      with Context(JIT=0): return model(t, v).realize()
    helper_test("test_gpt2", lambda: (Tensor([[1,]]),Variable("pos", 1, 100).bind(1)), test, 0.23, 168, all_jitted=True)

  @unittest.skipUnless(dtypes.float16 in supported_dtypes, "need dtypes.float16")
  def test_train_cifar_hyp(self):
    self.enterContext(Context(DEFAULT_FLOAT=dtypes.float16))
    with Context(TRAINING=1):
      model = SpeedyResNet(Tensor.ones((12,3,2,2)))
      optimizer = optim.SGD(get_parameters(model), lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
      initial_div_factor = hyp['opt']['initial_div_factor']
      final_lr_ratio = hyp['opt']['final_lr_ratio']
      pct_start = hyp['opt']['percent_start']
      lr_scheduler = OneCycleLR(optimizer, max_lr=hyp['opt']['bias_lr'], pct_start=pct_start, div_factor=initial_div_factor,
                                final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=4)
      assert not np.isnan(lr_scheduler.min_lr), "lr too small or initial_div_facotr too big for half"

if __name__ == '__main__':
  unittest.main()
