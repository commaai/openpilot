import gc
from tinygrad import Tensor, UOp, Device
from tinygrad.shape.shapetracker import views_to_indexed_uops
from tinygrad.engine.realize import method_cache, get_kernel

def uops_allocated(): return sum([isinstance(x, UOp) for x in gc.get_objects()])
def print_uops():
  for x in gc.get_objects():
    if isinstance(x, UOp): print(x)

def start(): pass
def single_tensor(): Tensor([2])
def two_plus_two(): Tensor([2])+Tensor([2])
def two_plus_two_schedule(): (Tensor([2])+Tensor([2])).schedule()
def two_plus_two_kernel():
  si = (Tensor([2])+Tensor([2])).schedule()[-1]
  get_kernel(Device.default.renderer, si.ast)
def two_plus_two_linearize():
  si = (Tensor([2])+Tensor([2])).schedule()[-1]
  k = get_kernel(Device.default.renderer, si.ast)
  k.get_optimized_ast()
  #k.linearize()
def two_plus_two_realize(): (Tensor([2])+Tensor([2])).realize()
def two_plus_two_item(): (Tensor([2])+Tensor([2])).item()
def gradient_test():
  x = Tensor.eye(3, requires_grad=True)
  y = Tensor([[2.0,0,-2.0]], requires_grad=True)
  z = y.matmul(x).sum()
  z.backward()
def realized_eye():
  Tensor.eye(3, requires_grad=True).realize()
def realized_list():
  Tensor([[2.0,0,-2.0]], requires_grad=True).realize()
def kernel_matmul():
  x = Tensor.eye(3, requires_grad=True)
  y = Tensor([[2.0,0,-2.0]], requires_grad=True)
  z = y.matmul(x)
  si = z.schedule()[-1]
  get_kernel(Device.default.renderer, si.ast)
def realized_matmul():
  x = Tensor.eye(3, requires_grad=True)
  y = Tensor([[2.0,0,-2.0]], requires_grad=True)
  z = y.matmul(x)
  Tensor.realize(z)
def realized_gradient():
  x = Tensor.eye(3, requires_grad=True)
  y = Tensor([[2.0,0,-2.0]], requires_grad=True)
  z = y.matmul(x).sum()
  z.backward()
  Tensor.realize(x, y, z, x.grad, y.grad)
tests = [start, single_tensor, two_plus_two, two_plus_two_schedule, two_plus_two_kernel,
         two_plus_two_linearize, two_plus_two_realize, two_plus_two_item, gradient_test,
         realized_eye, realized_list, kernel_matmul, realized_matmul, realized_gradient]

if __name__ == "__main__":
  gc.disable()
  start_uops = uops_allocated()
  # there's a few consts created as default values
  print_uops()
  for t in tests:
    t()

    # these caches will keep uops alive
    method_cache.clear()
    views_to_indexed_uops.cache_clear()

    new_uops = uops_allocated()
    gc.collect()
    new_uops_gc = uops_allocated()
    print(f"{t.__name__:30s}: {new_uops:3d} -> {new_uops_gc:3d}")
    assert new_uops == start_uops
  #print_uops()
