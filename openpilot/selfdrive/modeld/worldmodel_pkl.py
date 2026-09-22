import io
import pickle
import platform
import struct
from pathlib import Path

import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp

UPLOAD_CHUNK_SIZE = 32 << 20


def load_worldmodel(path: Path):
  device = Device[Device.DEFAULT]
  with path.open('rb') as f:
    opcodes = f.read(struct.unpack('<q', f.read(8))[0])
    size = path.stat().st_size - f.tell()
    arena = Buffer(Device.DEFAULT, round_up(size, UPLOAD_CHUNK_SIZE), dtypes.uint8, preallocate=True)
    class Unpickler(pickle.Unpickler):
      def persistent_load(self, pid):
        return arena.view(*pid)

    artifact = Unpickler(io.BytesIO(opcodes)).load()
    if artifact['arch'] != device.arch:
      raise RuntimeError(f"Worldmodel compiled for {artifact['arch']}, got {device.arch}")
    upload = artifact['programs'][platform.machine().lower()]['upload']
    staging = np.zeros(UPLOAD_CHUNK_SIZE, dtype=np.uint8)
    source = Tensor(staging, device='NPY')
    for offset in range(0, size, UPLOAD_CHUNK_SIZE):
      count = min(UPLOAD_CHUNK_SIZE, size - offset)
      assert f.readinto(memoryview(staging)[:count]) == count
      if count != UPLOAD_CHUNK_SIZE:
        staging[count:] = 0
      destination = Tensor(UOp.from_buffer(arena.view(UPLOAD_CHUNK_SIZE, dtypes.uint8, offset).ensure_allocated()))
      upload(destination, source)
      device.synchronize()
  return artifact
