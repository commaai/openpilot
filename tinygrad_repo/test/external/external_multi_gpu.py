#!/usr/bin/env python3
# cd extra/disassemblers/ && git clone --recursive github.com:geohot/cuda_ioctl_sniffer.git
# LD_PRELOAD=$PWD/extra/disassemblers/cuda_ioctl_sniffer/out/sniff.so GPU=1 python3 test/external/external_multi_gpu.py
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import colored, Timing, getenv
from tinygrad.device import Device

d0, d1 = f'{Device.DEFAULT}:0', f'{Device.DEFAULT}:1'

def sync():
  Device[d0].synchronize()
  Device[d1].synchronize()

if __name__ == "__main__":
  print("GPU devices", d0, d1)
  sz = getenv("N", 1024*1024*256)  # 1 GB

  with Timing("GPU initial sync: "): sync()

  with Timing("CPU creation: ", on_exit=lambda x: f", {(sz*4*2)/x:.2f} GB/sec"):
    c0 = (Tensor.ones(sz, device="CPU")/2).realize()
    c1 = (Tensor.ones(sz, device="CPU")/4).realize()
    print(c0.lazydata.base.realized)
    print(c1.lazydata.base.realized)

  with Timing("CPU -> 0: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    a0 = c0.to(d0).realize()
    sync()
  with Timing("CPU -> 1: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    b1 = c1.to(d1).realize()
    sync()

  # cross copy. this is (sometimes) going through the CPU
  with Timing("0 -> 1: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    a1 = a0.to(d1).realize()
    sync()
  with Timing("1 -> 0: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    b0 = b1.to(d0).realize()
    sync()

  # sum
  with Timing("0+0 -> 0 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    ab0 = (a0 + b0).realize()
    sync()
  with Timing("1+1 -> 1 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    ab1 = (a1 + b1).realize()
    sync()

  # cross device sum (does this work?)
  with Timing(colored("0+1 -> 0 (sum): ", "red"), on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    abx0 = (a0 + b1.to(d0)).realize()
    sync()

  with Timing(colored("1+0 -> 1 (sum): ", "red"), on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    abx1 = (b1 + a0.to(d1)).realize()
    sync()

  # copy back
  # NOTE: half of this slowness is caused by allocating memory on the CPU
  with Timing("0 -> CPU: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    cc0 = ab0.numpy()
  with Timing("1 -> CPU: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    cc1 = ab1.numpy()

  # same
  print("testing")
  np.testing.assert_allclose(cc0, cc1)

  # same (cross)
  print("testing (cross)")
  np.testing.assert_allclose(cc0, abx0.numpy())
  np.testing.assert_allclose(cc0, abx1.numpy())

  # devices
  print(ab0)
  print(ab1)
  print(abx0)
  print(abx1)
