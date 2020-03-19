import os
from cffi import FFI

# Workaround for the EON/termux build of Python having os.*xattr removed.
ffi = FFI()
ffi.cdef("""
int setxattr(const char *path, const char *name, const void *value, size_t size, int flags);
ssize_t getxattr(const char *path, const char *name, void *value, size_t size);
ssize_t listxattr(const char *path, char *list, size_t size);
""")
libc = ffi.dlopen(None)

def setxattr(path, name, value, flags=0):
  if libc.setxattr(path, name, value, len(value), flags) == -1:
    raise OSError(ffi.errno, os.strerror(ffi.errno))

def getxattr(path, name, size=128):
  value = ffi.new(f"char[{size}]")
  l = libc.getxattr(path, name, value, size)
  if l == -1:
    raise OSError(ffi.errno, os.strerror(ffi.errno))
  return ffi.buffer(value)[:l]

def listxattr(path, size=128):
  attrs = ffi.new(f"char[{size}]")
  l = libc.listxattr(path, attrs, size)
  if l == -1:
    raise OSError(ffi.errno, os.strerror(ffi.errno))
  # attrs is b'\0' delimited values (so chop off trailing empty item)
  return ffi.buffer(attrs)[:l].split(b"\0")[0:-1]
