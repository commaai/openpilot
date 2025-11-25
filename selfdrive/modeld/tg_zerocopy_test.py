#!/usr/bin/env python3
import time
import pickle
import numpy as np
from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from common.transformations.camera import get_nv12_info


YUV_SIZE = 5000000
a_np = (32*np.random.randn(YUV_SIZE).astype(np.float32) + 128).clip(0,255).astype(np.uint8)
a = Tensor.from_blob(a_np.ctypes.data, (YUV_SIZE,), dtype='uint8').realize()

print(a.numpy()[:10], a_np[:10])
assert np.all(a.numpy() == a_np), "Initial tensor data does not match numpy data"
assert np.all((a - 1).numpy() == a_np -1 ), "Initial tensor data does not match numpy data"
