from __future__ import annotations
import ctypes, os
try: import fcntl # windows misses that
except ImportError: fcntl = None #type:ignore[assignment]
from tinygrad.helpers import DEV, getenv, pluralize
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.memory import MMIOInterface as MMIOInterface, BumpAllocator as BumpAllocator

class FileIOInterface:
  """
  Hardware Abstraction Layer for HCQ devices. The class provides a unified interface for interacting with hardware devices.
  """

  def __init__(self, path:str="", flags:int=os.O_RDONLY, fd:int|None=None):
    self.path:str = path
    self.fd:int = fd or os.open(path, flags)
  def __del__(self):
    if hasattr(self, 'fd'): os.close(self.fd)
  def ioctl(self, request, arg): return fcntl.ioctl(self.fd, request, arg)
  def mmap(self, start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, self.fd, offset)
  def read(self, size=None, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as file: return file.read(size)
  def write(self, content, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as file: file.write(content)
  def listdir(self): return os.listdir(self.path)
  def seek(self, offset): os.lseek(self.fd, offset, os.SEEK_SET)
  @staticmethod
  def _mmap(start, sz, prot, flags, fd, offset):
    x = libc.mmap(start, sz, prot, flags, fd, offset)
    if x == 0xffffffffffffffff: raise OSError(f"Failed to mmap {sz} bytes at {hex(start)}: {os.strerror(ctypes.get_errno())}")
    return x
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, -1, offset)
  @staticmethod
  def munmap(buf, sz): return libc.munmap(buf, sz)
  @staticmethod
  def exists(path): return os.path.exists(path)
  @staticmethod
  def readlink(path): return os.readlink(path)
  @staticmethod
  def eventfd(initval, flags=None): return FileIOInterface(fd=os.eventfd(initval, flags))  # type: ignore[attr-defined]

if DEV.interface.startswith("MOCK"): from test.mockgpu.mockgpu import MockFileIOInterface as FileIOInterface  # noqa: F401 # pylint: disable=unused-import

# **************** for HCQ Compatible Devices ****************

def hcq_filter_visible_devices(devs, device):
  assert (v:=getenv("HCQ_VISIBLE_DEVICES", "")) == "", f"HCQ_VISIBLE_DEVICES={v} is deprecated, use DEV={DEV.target(device, indices=v)} instead"
  if '-' in (idstr:=DEV.target(device).indices): ids = list(range(int(idstr.split('-')[0]), int(idstr.split('-')[1])+1))
  else: ids = [int(x) for x in idstr.split(',') if x.strip()]
  assert all(x < len(devs) for x in ids), f"invalid visibility filter: {ids} ({pluralize('device', len(devs))} available)"
  return [devs[x] for x in ids] if ids else devs
