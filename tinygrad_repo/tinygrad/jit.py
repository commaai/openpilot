from typing import Callable, List, Tuple, Any, Dict, cast, Union, Optional
from collections import defaultdict
import functools, itertools
from tinygrad.helpers import DEBUG, DType, merge_dicts
from tinygrad.ops import RawBuffer, Device
from tinygrad.tensor import Tensor
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

JIT_SUPPORTED_DEVICE = ["GPU", "CLANG", "METAL", "CUDA", "HIP", "WEBGPU", "LLVM"]

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[Tuple[Any, List[Optional[RawBuffer]], Dict[Variable, int]]] = []
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], ShapeTracker, DType]]= {}   # (kernel_number, buffer_number) -> (input_name, expected_shapetracker, expected_type)
    self.updatable_entries: Dict[int, List[int]] = defaultdict(list) # (kernel_number) -> list(argument id). These are buffers from input + variables.

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT.split(":")[0] not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device
    # NOTE: this cast is needed since although we know realize will create a ".realized" RawBuffer, the type checker doesn't
    input_rawbuffers: Dict[Union[int, str], Tuple[RawBuffer, ShapeTracker]] = {cast(Union[int, str], k):(cast(RawBuffer, v.realize().lazydata.realized), v.lazydata.st) for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt >= 2:
      try: var_vals: Dict[Variable, int] = kwargs["jit_ctx"]
      except KeyError: var_vals = merge_dicts([arg.lazydata.st.var_vals for arg in args if arg.__class__ is Tensor])
      if len(var_vals) > 1: var_vals = dict(sorted(var_vals.items(), key=lambda kv: kv[0].key))
      for (j,i),(input_name, expected_st, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name][0].dtype == expected_type, f"type mismatch in JIT, {input_rawbuffers[input_name][0].dtype} != {expected_type}"
        # NOTE: if we pass jit_ctx instead of using reshape to update the var_vals, we cannot compare the shapetracker directly
        if "jit_ctx" not in kwargs: assert input_rawbuffers[input_name][1].unbind() == expected_st, f"ShapeTracker mismatch in JIT, {input_rawbuffers[input_name][1].unbind()} != {expected_st}"
        self.jit_cache[j][1][i] = input_rawbuffers[input_name][0]
      for j in self.updatable_entries.keys():
        for k in self.jit_cache[j][2].keys():
          try: self.jit_cache[j][2][k] = var_vals[k]
          except KeyError: pass
      for prg, pargs, variables in self.jit_cache: prg(pargs, variables, jit=True)
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 1:
      CacheCollector.start()
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = CacheCollector.finish()
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j_,cache in enumerate(self.jit_cache): # type: Tuple[int, Tuple[Callable, List[Optional[RawBuffer]], Dict[Variable, int]]]
        for i,a in enumerate(cache[1]):
          if a in [v[0] for v in input_rawbuffers.values()]:
            self.input_replace[(j_,i)] = [(k, v[1].unbind(), v[0].dtype) for k,v in input_rawbuffers.items() if v[0] == a][0]
            self.updatable_entries[j_].append(i)
        for i in range(len(cache[2])): self.updatable_entries[j_].append(len(cache[1])+i)
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt += 1
    return self.ret

class _CacheCollector:
  def __init__(self): self.cache: Optional[List[Tuple[Callable, List[Any], Dict[Any,Any]]]] = None
  def start(self): self.cache = []
  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    self.cache.append((prg, rawbufs, var_vals))
  def finish(self):
    if self.cache is None: return []
    ret = self.cache
    self.cache = None
    return ret
CacheCollector = _CacheCollector()
