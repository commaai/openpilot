import numpy as np
from tinygrad import Device, Tensor, dtypes


def upload(value, dtype=dtypes.bfloat16):
  source = value if isinstance(value, Tensor) else Tensor(np.asarray(value), device='CPU')
  return source.to(Device.DEFAULT).cast(dtype).contiguous().realize()


def download(value):
  return value.float().numpy()
