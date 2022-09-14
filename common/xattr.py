import os
from cffi import FFI
from typing import Any, List

# Workaround for the EON/termux build of Python having os.*xattr removed.
ffi = FFI()
ffi.cdef("""
int setxattr(const char *path, const char *name, const void *value, size_t size, int flags);
ssize_t getxattr(const char *path, const char *name, void *value, size_t size);
ssize_t listxattr(const char *path, char *list, size_t size);
int removexattr(const char *path, const char *name);
""")
libc = ffi.dlopen(None)

def setxattr(path, name, value, flags=0) -> None:
  path = path.encode()
  name = name.encode()
  if libc.setxattr(path, name, value, len(value), flags) == -1:
    raise OSError(ffi.errno, f"{os.strerror(ffi.errno)}: setxattr({path}, {name}, {value}, {flags})")

def getxattr(path, name, size=128):
  path = path.encode()
  name = name.encode()
  value = ffi.new(f"char[{size}]")
  l = libc.getxattr(path, name, value, size)
  if l == -1:
    # errno 61 means attribute hasn't been set
    if ffi.errno == 61:
      return None
    raise OSError(ffi.errno, f"{os.strerror(ffi.errno)}: getxattr({path}, {name}, {size})")
  return ffi.buffer(value)[:l]

def listxattr(path, size=128) -> List[Any]:
  path = path.encode()
  attrs = ffi.new(f"char[{size}]")
  l = libc.listxattr(path, attrs, size)
  if l == -1:
    raise OSError(ffi.errno, f"{os.strerror(ffi.errno)}: listxattr({path}, {size})")
  # attrs is b'\0' delimited values (so chop off trailing empty item)
  return [a.decode() for a in ffi.buffer(attrs)[:l].split(b"\0")[0:-1]]

def removexattr(path, name) -> None:
  path = path.encode()
  name = name.encode()
  if libc.removexattr(path, name) == -1:
    raise OSError(ffi.errno, f"{os.strerror(ffi.errno)}: removexattr({path}, {name})")
