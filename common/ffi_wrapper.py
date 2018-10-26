import os
import sys
import fcntl
import hashlib
from cffi import FFI

TMPDIR = "/tmp/ccache"

def ffi_wrap(name, c_code, c_header, tmpdir=TMPDIR, cflags="", libraries=[]):
  cache = name + "_" + hashlib.sha1(c_code).hexdigest()
  try:
    os.mkdir(tmpdir)
  except OSError:
    pass

  fd = os.open(tmpdir, 0)
  fcntl.flock(fd, fcntl.LOCK_EX)
  try:
    sys.path.append(tmpdir)
    try:
      mod = __import__(cache)
    except Exception:
      print "cache miss", cache
      compile_code(cache, c_code, c_header, tmpdir, cflags, libraries)
      mod = __import__(cache)
  finally:
    os.close(fd)

  return mod.ffi, mod.lib

def compile_code(name, c_code, c_header, directory, cflags="", libraries=[]):
  ffibuilder = FFI()
  ffibuilder.set_source(name, c_code, source_extension='.cpp', libraries=libraries)
  ffibuilder.cdef(c_header)
  os.environ['OPT'] = "-fwrapv -O2 -DNDEBUG -std=c++11"
  os.environ['CFLAGS'] = cflags
  ffibuilder.compile(verbose=True, debug=False, tmpdir=directory)

def wrap_compiled(name, directory):
  sys.path.append(directory)
  mod = __import__(name)
  return mod.ffi, mod.lib
