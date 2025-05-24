# NOTE: this only tests the speed of the LLaMA codegen, it doesn't actually run the net
import unittest, time
from examples.llama import Transformer, MODEL_PARAMS
from tinygrad.tensor import Tensor
from tinygrad import Device
from tinygrad.nn.state import get_state_dict
from tinygrad.device import Allocator, Compiled
from tinygrad.engine.realize import method_cache
from tinygrad.helpers import Profiling

class FakeProgram:
  def __init__(self, name:str, prg:bytes): pass
  def __call__(self, *bufs, global_size, local_size, vals=(), wait=False): pass

class FakeAllocator(Allocator[Compiled]):
  def _alloc(self, sz, options): return None
  def _copyin(self, dest, src:memoryview): pass

class TestLLaMASpeed(unittest.TestCase):
  def test_llama_compile(self):
    backup_program = Device[Device.DEFAULT].runtime
    backup_allocator = Device[Device.DEFAULT].allocator
    backup_compiler = Device[Device.DEFAULT].compiler
    Device[Device.DEFAULT].runtime = FakeProgram
    Device[Device.DEFAULT].allocator = FakeAllocator(Device.default)

    print("testing llama python run time")
    model = Transformer(**MODEL_PARAMS["1"]["7B"]["args"])
    print("built model")
    # assign fake tensors to the values
    for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype))
    print("assigned empty tensors, doing warmup")

    def run_llama(st, empty_method_cache=True):
      if empty_method_cache: method_cache.clear()
      tms = [time.perf_counter()]
      for i in range(5):
        model(Tensor([[1,2,3,4]]), i).realize()
        tms.append(time.perf_counter())
      timings = [(tms[i+1]-tms[i])*1000 for i in range(len(tms)-1)]
      print(f"{st:15s} mean runtime: {sum(timings)/len(timings):7.2f}ms, runs: ", ", ".join(f'{x:7.2f}' for x in timings))

    run_llama("codegen(0)")
    run_llama("codegen(1)")

    # test no compiler use for this
    Device[Device.DEFAULT].compiler = None
    run_llama("methodcache", False)
    with Profiling(sort='time', frac=0.1, fn="/tmp/llama.prof", ts=5):
      run_llama("profile", False)

    Device[Device.DEFAULT].runtime = backup_program
    Device[Device.DEFAULT].allocator = backup_allocator
    Device[Device.DEFAULT].compiler = backup_compiler

if __name__ == '__main__':
  TestLLaMASpeed().test_llama_compile()
  #unittest.main()
