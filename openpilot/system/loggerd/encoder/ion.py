"""ctypes interface to the legacy msm ION allocator, port of msgq/visionipc/visionbuf_ion.cc.

Constants and struct layouts were extracted from the agnos-kernel-sdm845 UAPI headers
(linux/ion.h, linux/msm_ion.h).
"""
import ctypes
import mmap
import os
import threading

from openpilot.system.loggerd.encoder.v4l2 import _IOWR, safe_ioctl, u32

ION_DEVICE = "/dev/ion"


class ion_allocation_data(ctypes.Structure):
  _fields_ = [
    ("len", ctypes.c_size_t),
    ("align", ctypes.c_size_t),
    ("heap_id_mask", u32),
    ("flags", u32),
    ("handle", ctypes.c_int),
  ]


class ion_fd_data(ctypes.Structure):
  _fields_ = [
    ("handle", ctypes.c_int),
    ("fd", ctypes.c_int),
  ]


class ion_handle_data(ctypes.Structure):
  _fields_ = [("handle", ctypes.c_int)]


class ion_custom_data(ctypes.Structure):
  _fields_ = [
    ("cmd", ctypes.c_uint),
    ("arg", ctypes.c_ulong),
  ]


class ion_flush_data(ctypes.Structure):
  _fields_ = [
    ("handle", ctypes.c_int),
    ("fd", ctypes.c_int),
    ("vaddr", ctypes.c_void_p),
    ("offset", u32),
    ("length", u32),
  ]


assert ctypes.sizeof(ion_allocation_data) == 32
assert ctypes.sizeof(ion_fd_data) == 8
assert ctypes.sizeof(ion_handle_data) == 4
assert ctypes.sizeof(ion_custom_data) == 16 and ion_custom_data.arg.offset == 8
assert ctypes.sizeof(ion_flush_data) == 24 and ion_flush_data.vaddr.offset == 8

ION_IOC_ALLOC = _IOWR('I', 0, ion_allocation_data)
ION_IOC_FREE = _IOWR('I', 1, ion_handle_data)
ION_IOC_SHARE = _IOWR('I', 4, ion_fd_data)
ION_IOC_CUSTOM = _IOWR('I', 6, ion_custom_data)

ION_IOC_CLEAN_CACHES = _IOWR('M', 0, ion_flush_data)
ION_IOC_INV_CACHES = _IOWR('M', 1, ion_flush_data)

assert ION_IOC_ALLOC == 0xc0204900
assert ION_IOC_FREE == 0xc0044901
assert ION_IOC_SHARE == 0xc0084904
assert ION_IOC_CUSTOM == 0xc0104906
assert ION_IOC_CLEAN_CACHES == 0xc0184d00
assert ION_IOC_INV_CACHES == 0xc0184d01

ION_IOMMU_HEAP_ID = 25
ION_FLAG_CACHED = 1

_ion_fd = None
_ion_fd_lock = threading.Lock()


def ion_fd() -> int:
  global _ion_fd
  with _ion_fd_lock:
    if _ion_fd is None:
      _ion_fd = os.open(ION_DEVICE, os.O_RDWR | os.O_NONBLOCK)
  return _ion_fd


class IonBuf:
  """A cached, mmapped DMA buffer from the ION system heap."""

  def __init__(self, length: int):
    alloc = ion_allocation_data(len=length, align=4096, heap_id_mask=1 << ION_IOMMU_HEAP_ID, flags=ION_FLAG_CACHED)
    safe_ioctl(ion_fd(), ION_IOC_ALLOC, alloc)
    self.handle = alloc.handle

    fd_data = ion_fd_data(handle=alloc.handle)
    safe_ioctl(ion_fd(), ION_IOC_SHARE, fd_data)
    self.fd = fd_data.fd

    self.len = length
    self.mm = mmap.mmap(self.fd, length)
    self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
    ctypes.memset(self.addr, 0, length)

  def sync_from_device(self) -> None:
    # invalidate the CPU cache before reading what the device wrote (~ DMA_FROM_DEVICE)
    flush = ion_flush_data(handle=self.handle, vaddr=self.addr, offset=0, length=self.len)
    custom = ion_custom_data(cmd=ION_IOC_INV_CACHES, arg=ctypes.addressof(flush))
    safe_ioctl(ion_fd(), ION_IOC_CUSTOM, custom)

  def free(self) -> None:
    self.mm.close()
    os.close(self.fd)
    safe_ioctl(ion_fd(), ION_IOC_FREE, ion_handle_data(handle=self.handle))
