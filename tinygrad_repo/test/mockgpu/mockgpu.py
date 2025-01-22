import ctypes, ctypes.util, struct, platform, time, os, builtins, atexit
from test.mockgpu.nv.nvdriver import NVDriver
from test.mockgpu.amd.amddriver import AMDDriver
from tinygrad.helpers import to_mv
start = time.perf_counter()

# *** ioctl lib ***
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int
libc.fdopendir.argtypes = [ctypes.c_int]
libc.fdopendir.restype = ctypes.c_void_p

# platform.processor calls `uname -p` which can return `unknown` on some systems
processor = os.getenv("IOCTL_PROCESSOR") or platform.processor()
OPEN_SYSCALL = {"aarch64": None, "x86_64": 2}[processor]
CLOSE_SYSCALL = {"aarch64": 57, "x86_64": 3}[processor]
READ_SYSCALL = {"aarch64": 63, "x86_64": 0}[processor]
IOCTL_SYSCALL = {"aarch64": 29, "x86_64": 16}[processor]
MMAP_SYSCALL = {"aarch64": 222, "x86_64": 9}[processor]
LSEEK_SYSCALL = {"aarch64": 62, "x86_64": 8}[processor]
NEWFSTATAT_SYSCALL = {"aarch64": 79, "x86_64": 262}[processor]
GETDENTS64_SYSCALL = {"aarch64": 61, "x86_64": 217}[processor]

def install_hook(c_function, python_function):
  python_function_addr = ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value
  if processor == "x86_64":
    # tramp = b"\x49\xB8" + struct.pack("Q", python_function_addr) + b"\x41\xFF\xE0"
    # push r9
    # push r9
    # mov r9, 0x1122334455667788
    # mov [rsp+8], r9
    # pop r9
    # ret
    tramp = b"\x41\x51\x41\x51\x49\xB9" + struct.pack("Q", python_function_addr) + b"\x4C\x89\x4C\x24\x08\x41\x59\xC3"
  else:
    raise Exception(f"processor {processor} not supported")

  original_bc = (ctypes.c_char * 64)()

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  libc.memcpy(original_bc, ioctl_address.contents, len(tramp))
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))

  # Restore correct functions to close libs after python exits
  def __restore(): libc.memcpy(ioctl_address.contents, original_bc, len(tramp))
  atexit.register(__restore)

drivers = [AMDDriver(), NVDriver()]
tracked_fds = {}

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_ulong)
def _open(name, flags, mode):
  for d in drivers:
    pyname = name.decode()
    for x in d.tracked_files:
      if pyname == x.path:
        virtfd = d.open(pyname, flags, mode, x)
        tracked_fds[virtfd.fd] = virtfd
        return virtfd.fd

  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_int, ctypes.c_ulong]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(OPEN_SYSCALL, name, flags, mode)

@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)
def _opendir(name):
  fd = _open(name, os.O_RDONLY| os.O_DIRECTORY, 0)
  if fd >= 0x80:
    fake_dirfd = _open(".".encode(), os.O_RDONLY| os.O_DIRECTORY, 0)
    st = libc.fdopendir(fake_dirfd)
    to_mv(st, 8).cast('Q')[0] = fd
    return st
  else: return libc.fdopendir(fd)

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
def _close(fd):
  if fd in tracked_fds:
    tracked_fds[fd].close(fd)
    tracked_fds.pop(fd)
    return 0

  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_int]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(CLOSE_SYSCALL, fd)

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
def _closedir(st): return _close(to_mv(st, 8).cast('Q')[0])

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def _ioctl(fd, request, argp):
  if fd in tracked_fds: return tracked_fds[fd].ioctl(fd, request, argp)

  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(IOCTL_SYSCALL, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))

@ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
def _read(fd, buf, sz):
  if fd in tracked_fds: return tracked_fds[fd].read(fd, buf, sz)

  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(READ_SYSCALL, ctypes.c_int(fd), ctypes.c_void_p(buf), ctypes.c_size_t(sz))

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_int)
def _lseek64(fd, off, whence):
  if fd in tracked_fds: return tracked_fds[fd].lseek(fd, off, whence)

  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong, ctypes.c_int]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(LSEEK_SYSCALL, fd, off, whence)

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def _stat64(name, buf):
  for d in drivers:
    pyname = name.decode()
    for x in d.tracked_files:
      if pyname == x.path:
        virtfd = d.open(pyname, 0, 0, x)
        return virtfd.fstat(virtfd.fd, buf)

  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_ulong]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(NEWFSTATAT_SYSCALL, -100, name, ctypes.c_void_p(buf), 0)

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
def _fstat64(fd, buf):
  if fd in tracked_fds: return tracked_fds[fd].fstat(fd, buf)

  empty_str = (ctypes.c_char*1)()
  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_ulong]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(NEWFSTATAT_SYSCALL, ctypes.c_int(fd), empty_str, ctypes.c_void_p(buf), 0x1000)

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong)
def _getdents64(fd, buf, sz):
  if fd in tracked_fds: return tracked_fds[fd].getdents(fd, buf, sz)

  libc.syscall.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
  libc.syscall.restype = ctypes.c_int
  return libc.syscall(GETDENTS64_SYSCALL, fd, buf, sz)

def _mmap(start, sz, prot, flags, fd, offset):
  if fd in tracked_fds: return tracked_fds[fd].mmap(start, sz, prot, flags, fd, offset)
  return libc.mmap(start, sz, prot, flags, fd, offset)

def _munmap(buf, sz):
  return libc.munmap(buf, sz)

orignal_memoryview = builtins.memoryview
class TrackedMemoryView:
  def __init__(self, data, rcb, wcb):
    self.mv = orignal_memoryview(data)
    self.rcb, self.wcb = rcb, wcb

  def __getitem__(self, index):
    self.rcb(self.mv, index)
    return self.mv[index]

  def __setitem__(self, index, value):
    self.mv[index] = value
    self.wcb(self.mv, index)

  def cast(self, new_type, **kwargs):
    self.mv = self.mv.cast(new_type, **kwargs)
    return self

  @property
  def nbytes(self): return self.mv.nbytes
  def __len__(self): return len(self.mv)
  def __repr__(self): return repr(self.mv)

def _memoryview(cls, mem):
  if isinstance(mem, int) or isinstance(mem, ctypes.Array):
    addr = ctypes.addressof(mem) if isinstance(mem, ctypes.Array) else mem
    for d in drivers:
      for st,en,rcb,wcb in d.tracked_addresses:
        if st <= addr <= en: return TrackedMemoryView(mem, rcb, wcb)
  return orignal_memoryview(mem)

install_hook(libc.open, _open)
install_hook(libc.opendir, _opendir)
install_hook(libc.close, _close)
install_hook(libc.closedir, _closedir)
install_hook(libc.ioctl, _ioctl)
install_hook(libc.read, _read)
install_hook(libc.lseek64, _lseek64)
install_hook(libc.stat64, _stat64)
install_hook(libc.fstat64, _fstat64)
install_hook(libc.getdents64, _getdents64)
builtins.memoryview = type("memoryview", (), {'__new__': _memoryview}) # type: ignore

# rewrite autogen's libc mmaps functions.
import tinygrad.runtime.autogen.libc as autogen_libc
autogen_libc.mmap = _mmap # type: ignore
autogen_libc.munmap = _munmap # type: ignore
