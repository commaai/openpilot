from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Timing

N = 512
GPUS = 5
ds = tuple([f"{Device.DEFAULT}:{i+1}" for i in range(GPUS)])
t = [Tensor.ones(N, N, N, device=d).contiguous().realize() for d in ds]

for _ in range(10):
  GlobalCounters.reset()
  with Timing():
    for ti in t:
      ti.to_(ds[(ds.index(ti.device)+1+len(ds))%len(ds)])
      # ti.to_(ds[(ds.index(ti.device)-1+len(ds))%len(ds)]) # reversed order
      ti.realize()