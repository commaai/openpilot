#!/usr/bin/env python3
"""
Stress test for beam timeout + device recovery on AM devices.

Usage:
  DEV=AMD python test/external/external_fuzz_beam_timeout_recovery.py
"""
from tinygrad import Tensor, Device
from tinygrad.helpers import Context
from tinygrad.runtime.ops_amd import AMDDevice

if __name__ == "__main__":
  dev = Device["AMD"]
  assert isinstance(dev, AMDDevice) and dev.is_am(), "not am"

  N = 10000
  for i in range(N):
    with Context(DEBUG=0, BEAM=0):
      a = Tensor.rand(4096, 4096, device="AMD").contiguous().realize()
      b = Tensor.rand(4096, 4096, device="AMD").contiguous().realize()
      c = a.matmul(b)
      c.realize()
    try: dev.synchronize(timeout=1)
    except RuntimeError as e: print(e)
    with Context(DEBUG=0, BEAM=0):
      a = Tensor.ones(512, 512, device="AMD").contiguous().realize()
      b = Tensor.ones(512, 512, device="AMD").contiguous().realize()
      result = a.matmul(b).realize()[0, 0].item()
    assert result == 512.0, f"iter {i}: got {result}"
    print(f"  iter {i+1}/{N}: ok")
  print(f"=== All {N} iterations passed ===")
