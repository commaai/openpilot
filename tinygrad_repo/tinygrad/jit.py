from typing import Callable, List, Tuple, Any, Dict, cast, Union
import itertools
from tinygrad.helpers import DEBUG, colored

from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.ops import GlobalCounters, CompiledBuffer, RawBuffer

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn : Callable = fxn
    self.cnt : int = 0
    self.jit_cache : List[Tuple[Callable, Any]] = []  # TODO: Any should be List[RawBuffer], but this fails
    self.ret : Any = None
    self.input_replace : Dict[Tuple[int, int], Union[int, str]]= {}

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT not in ["GPU", "CLANG", "METAL", "CUDA"]: return self.fxn(*args, **kwargs)  # only jit on the GPU codegen
    # NOTE: this cast is needed since although we know realize will create a ".realized" DeviceBuffer, the type checker doesn't
    input_rawbuffers : Dict[Union[int, str], RawBuffer] = {cast(Union[int, str], k):cast(CompiledBuffer, v.realize().lazydata.realized).raw() for k,v in itertools.chain(enumerate(args), kwargs.items()) if isinstance(v, Tensor)}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    if self.cnt >= 2:
      for (j,i),idx in self.input_replace.items(): self.jit_cache[j][1][i] = input_rawbuffers[idx]
      if DEBUG >= 2: print(colored("TOK", 'magenta').split("TOK")[0], end="")
      for prg, args in self.jit_cache: prg(args)
      if DEBUG >= 2: print(colored("TOK", 'magenta').split("TOK")[1], end="")
      for (j,i),idx in self.input_replace.items(): self.jit_cache[j][1][i] = None
    elif self.cnt == 1:
      GlobalCounters.cache = []
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = GlobalCounters.cache
      GlobalCounters.cache = None
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j,(prg,args) in enumerate(self.jit_cache):  # pylint: disable=E1133
        for i,a in enumerate(args):
          if a in input_rawbuffers.values():
            self.input_replace[(j,i)] = [k for k,v in input_rawbuffers.items() if v == a][0]
      assert set(self.input_replace.values()) == set(input_rawbuffers.keys()), "some input tensors not found"
      for (j,i),idx in self.input_replace.items(): self.jit_cache[j][1][i] = None
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt += 1
    return self.ret
