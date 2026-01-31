from tinygrad.device import Device

if Device.DEFAULT == "AMD":
  WARP_THREADS = 64
else:
  WARP_THREADS = 32
