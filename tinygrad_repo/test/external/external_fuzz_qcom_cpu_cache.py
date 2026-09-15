#!/usr/bin/env python3
from tinygrad.tensor import Tensor
import numpy as np

while True:
  arr = np.ones(1000000, dtype=np.uint8)
  print(f"numpy: {(arr + 1)[:10]}")

  ptr = arr.ctypes.data
  tensor = Tensor.from_blob(ptr, arr.shape, dtype='uint8', device='QCOM').realize() + 1
  print(f"from_blob: {tensor.numpy()[:10]}")
