# NOTE: this is probably the wrong backend to work on, you want backend.py
# discussion about this in #pytorch-backend on our Discord
# this is using high level debug features from torch and we should integrate deeper

from tinygrad import Tensor
import torch, contextlib
from torch.utils._python_dispatch import TorchDispatchMode
from tinygrad.dtype import _from_torch_dtype

def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  return TTensor(Tensor.empty(*size, dtype=_from_torch_dtype(dtype)))

# NOTE: if we have a way to change wrap/unwrap, these can be the same methods from backend.py
tiny_backend = {
  "aten.empty.memory_format": empty_memory_format,
  "aten.view.default": lambda x,sz: TTensor(x.tiny.reshape(sz)),
  "aten.abs.default": lambda x: TTensor(x.tiny.abs()),
  "aten.eq.Tensor": lambda x,y: TTensor(x.tiny == y.tiny),
  "aten.bitwise_and.Tensor": lambda x,y: TTensor(x.tiny & y.tiny),
  "aten.ne.Scalar": lambda x,y: TTensor(x.tiny != y),
  "aten.mul.Tensor": lambda x,y: TTensor(x.tiny * y.tiny),
  "aten.masked_select.default": lambda x,y: TTensor(Tensor(x.tiny.numpy()[y.tiny.numpy()])),
}

class TTensor(torch.Tensor):
  tiny: Tensor
  context = contextlib.nullcontext

  @staticmethod
  def __new__(cls, tiny, *args, **kwargs):
    out = torch.Tensor._make_wrapper_subclass(cls, tiny.shape)
    torch._C._set_throw_on_mutable_data_ptr(out)
    out.tiny = tiny
    return out
  def __repr__(self): return super().__repr__(tensor_contents=f"{self.tiny}")
  def __torch_dispatch__(cls, func, types, args, kwargs=None):
    print(f"Dispatch Log: {func}(*{[type(x) for x in args]}, **{kwargs.keys()})")
    #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
    new_func = tiny_backend.get(str(func), None)
    if new_func is None: raise NotImplementedError(f"add support for {func}")
    return new_func(*args, **(kwargs or {}))

class Dispatcher(TorchDispatchMode): __torch_dispatch__ = TTensor.__torch_dispatch__
Dispatcher().__enter__()

if __name__ == "__main__":
  a = torch.empty((4,), dtype=torch.int)
