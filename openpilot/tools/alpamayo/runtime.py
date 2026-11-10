import os

import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp


def upload(value, dtype=dtypes.bfloat16):
  source = value if isinstance(value, Tensor) else Tensor(np.asarray(value), device='CPU')
  if not int(os.getenv('ALPAMAYO_USB_PCIE', '0')):
    return source.to(Device.DEFAULT).cast(dtype).contiguous().realize()
  dev = Device[Device.DEFAULT]
  assert dev.is_usb, '--usb-pcie requires Chestnut'
  raw = source.to('CPU').cast(dtype).contiguous().bitcast(dtypes.uint8).data().tobytes()
  buffer = Buffer(Device.DEFAULT, round_up(len(raw), 4), options=BufferSpec(cpu_access=True)).ensure_allocated()
  dev.synchronize()
  buffer.host.view(size=buffer.nbytes, fmt='B')[:] = raw.ljust(buffer.nbytes, b'\0')
  return Tensor(UOp.from_buffer(buffer, dtype))[:source.numel()].reshape(source.shape)


def download(value):
  if not int(os.getenv('ALPAMAYO_USB_PCIE', '0')):
    return value.float().numpy()
  value = value.float().contiguous().realize()
  dev = Device[value.device]
  dev.synchronize()
  buffer = value._buffer()
  offset, remaining = buffer.offset, buffer.nbytes
  chunks = []
  for address, size in buffer.base.get_storage().meta.mapping.paddrs:
    if offset >= size:
      offset -= size
      continue
    take = min(remaining, size-offset)
    for start in range(0, take, 65536):
      count = min(take-start, 65536)
      view = dev.iface.pci_dev.map_bar(bar=dev.iface.vram_bar, off=address+offset+start, size=count)
      chunks.append(bytes(view[:]))
    remaining -= take
    offset = 0
    if not remaining:
      break
  assert remaining == 0
  return np.frombuffer(b''.join(chunks), dtype=np.float32).reshape(value.shape)
