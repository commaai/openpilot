import numpy as np

# Python version of openpilot/system/camerad/cameras/nv12_info.h
# Calculations from media/msm_media_info.h (VENUS_BUFFER_SIZE)

def align(val: int, alignment: int) -> int:
  return ((val + alignment - 1) // alignment) * alignment

def get_nv12_info(width: int, height: int) -> tuple[int, int, int, int]:
  """Returns (stride, y_height, uv_height, buffer_size) for NV12 frame dimensions."""
  stride = align(width, 128)
  y_height = align(height, 32)
  uv_height = align(height // 2, 16)

  # VENUS_BUFFER_SIZE for NV12
  y_plane = stride * y_height
  uv_plane = stride * uv_height + 4096
  size = y_plane + uv_plane + max(16 * 1024, 8 * stride)
  size = align(size, 4096)
  size += align(width, 512) * 512  # kernel padding for non-aligned frames
  size = align(size, 4096)

  return stride, y_height, uv_height, size


def nv12_planes(buf):
  data = np.frombuffer(buf.data, dtype=np.uint8)
  y = data[:buf.stride * buf.height].reshape(buf.height, buf.stride)[:, :buf.width]
  uv = data[buf.uv_offset:buf.uv_offset + buf.stride * (buf.height // 2)].reshape(-1, buf.stride)[:, :buf.width]
  return y, uv[:, ::2], uv[:, 1::2]
