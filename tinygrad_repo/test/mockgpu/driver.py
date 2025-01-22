import ctypes, struct, os
from typing import Any
from dataclasses import dataclass
from tinygrad.helpers import round_up

class VirtFileDesc:
  def __init__(self, fd): self.fd, self.off = fd, 0
  def read(self, fd, buf, sz): raise NotImplementedError()
  def ioctl(self, fd, req, argp): raise NotImplementedError()
  def mmap(self, st, sz, prot, flags, fd, off): raise NotImplementedError()
  def write(self, fd, buf, sz): raise NotImplementedError()
  def lseek(self, fd, off, whence): raise NotImplementedError()
  def fstat(self, fd, buf): raise NotImplementedError()
  def getdents(self, fd, buf, sz): return -1
  def close(self, fd): return 0

class TextFileDesc(VirtFileDesc):
  def __init__(self, fd, text):
    super().__init__(fd)
    self.content = ctypes.create_string_buffer(text.encode())
    self.sz = len(self.content) - 1

  def ioctl(self, fd, req, argp): return 0
  def write(self, fd, buf, sz): return -1
  def read(self, fd, buf, sz):
    ctypes.memmove(buf, ctypes.addressof(self.content) + self.off, rdsz:=min(sz, self.sz - self.off))
    self.off += rdsz
    return rdsz
  def lseek(self, fd, off, whence):
    if whence == os.SEEK_SET: self.off = off
    elif whence == os.SEEK_CUR: self.off += off
    elif whence == os.SEEK_END: self.off = self.sz + off
    else: return -1
    return 0
  def fstat(self, fd, buf):
    ctypes.memmove(buf, VirtFile.build_fstat(st_size=self.sz), 88)
    return 0

class DirFileDesc(VirtFileDesc):
  def __init__(self, fd, child_names):
    super().__init__(fd)
    child_names = ['.', '..'] + child_names

    tmp = b''
    for ino, name in enumerate(child_names):
      tmp += VirtFile.build_dirent(ino + 1, 0, name)
    self.content = ctypes.create_string_buffer(tmp)
    self.sz = len(self.content) - 1

  def ioctl(self, fd, req, argp): return 0
  def write(self, fd, buf, sz): return -1
  def read(self, fd, buf, sz): return -1
  def lseek(self, fd, off, whence):
    if whence == os.SEEK_SET: self.off = off
    elif whence == os.SEEK_CUR: self.off += off
    elif whence == os.SEEK_END: self.off = self.sz + off
    else: return -1
    return 0

  def getdents(self, fd, buf, sz):
    if self.sz == self.off: return 0
    if sz < self.sz: return -1
    ctypes.memmove(buf, ctypes.addressof(self.content) + self.off, self.sz)
    self.off = self.sz
    return self.sz

  def fstat(self, fd, buf):
    ctypes.memmove(buf, VirtFile.build_fstat(st_mode=0o40755), 96)
    return 0

@dataclass(frozen=True)
class VirtFile:
  path: str
  fdcls: Any # TODO: fix this Union[VirtFileDesc, functools.partial[VirtFileDesc]]

  @staticmethod
  def build_fstat(st_dev=0x20, st_ino=0x100000, st_mode=0o100777, st_nlink=1, st_uid=0, st_gid=0, st_rdev=0, st_size=0,
                  st_blksize=4096, st_blocks=0, st_atime=0, st_mtime=0, st_ctime=0):
    fmt_string = 'QQQIIIQQiQqqq'
    ssz = struct.calcsize(fmt_string)
    assert ssz == 96, f"{ssz} != 96"
    return struct.pack(fmt_string, st_dev, st_ino, st_nlink, st_mode, st_uid, st_gid,
                       st_rdev, st_size, st_blksize, st_blocks, st_atime, st_mtime, st_ctime)

  @staticmethod
  def build_dirent(d_ino, d_off, d_name, d_type=None):
    # Start with packing inode number, offset, and record length
    d_reclen = round_up(19 + len(d_name) + 1, 8)
    packed_data = struct.pack('QQHc', d_ino, d_off, d_reclen, b'\x04')
    d_name_bytes = d_name.encode()
    return packed_data + d_name_bytes + b'\x00' + b'\x00' * (d_reclen - (19 + len(d_name) + 1))

class VirtDriver:
  def __init__(self):
    self.tracked_files = []
    self.tracked_addresses = []
  def track_address(self, staddr, enaddr, rcb, wcb): self.tracked_addresses.append((staddr, enaddr, rcb, wcb))
  def open(self, name, flags, mode, fdcls): raise NotImplementedError()
