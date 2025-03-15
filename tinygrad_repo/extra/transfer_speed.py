from tinygrad import Tensor, Device

#N = 1024
N = 32
t = Tensor.rand(N, N, N, device="CPU").realize()
d1 = Device.DEFAULT + ":1"
d2 = Device.DEFAULT + ":2"
d3 = Device.DEFAULT + ":3"

for i in range(3):
  t.to_(d1)
  t.realize()
  # t.to_("CPU")
  # t.realize()
  t.to_(d2)
  t.realize()
  t.to_(d3)
  t.realize()
