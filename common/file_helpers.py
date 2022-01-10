import os
import shutil
import tempfile
from atomicwrites import AtomicWriter


def mkdirs_exists_ok(path):
  if path.startswith('http://') or path.startswith('https://'):
    raise ValueError('URL path')
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise


def rm_not_exists_ok(path):
  try:
    os.remove(path)
  except OSError:
    if os.path.exists(path):
      raise


def rm_tree_or_link(path):
  if os.path.islink(path):
    os.unlink(path)
  elif os.path.isdir(path):
    shutil.rmtree(path)


def get_tmpdir_on_same_filesystem(path):
  normpath = os.path.normpath(path)
  parts = normpath.split("/")
  if len(parts) > 1 and parts[1] == "scratch":
    return "/scratch/tmp"
  elif len(parts) > 2 and parts[2] == "runner":
    return f"/{parts[1]}/runner/tmp"
  return "/tmp"


class NamedTemporaryDir():
  def __init__(self, temp_dir=None):
    self._path = tempfile.mkdtemp(dir=temp_dir)

  @property
  def name(self):
    return self._path

  def close(self):
    shutil.rmtree(self._path)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


class CallbackReader:
  """Wraps a file, but overrides the read method to also
  call a callback function with the number of bytes read so far."""
  def __init__(self, f, callback, *args):
    self.f = f
    self.callback = callback
    self.cb_args = args
    self.total_read = 0

  def __getattr__(self, attr):
    return getattr(self.f, attr)

  def read(self, *args, **kwargs):
    chunk = self.f.read(*args, **kwargs)
    self.total_read += len(chunk)
    self.callback(*self.cb_args, self.total_read)
    return chunk


def _get_fileobject_func(writer, temp_dir):
  def _get_fileobject():
    return writer.get_fileobject(dir=temp_dir)
  return _get_fileobject

def monkeypatch_os_link():
  # This is neccesary on EON/C2, where os.link is patched out of python
  if not hasattr(os, 'link'):
    from cffi import FFI
    ffi = FFI()
    ffi.cdef("int link(const char *oldpath, const char *newpath);")
    libc = ffi.dlopen(None)

    def link(src, dest):
      return libc.link(src.encode(), dest.encode())
    os.link = link

def atomic_write_on_fs_tmp(path, **kwargs):
  """Creates an atomic writer using a temporary file in a temporary directory
     on the same filesystem as path.
  """
  # TODO(mgraczyk): This use of AtomicWriter relies on implementation details to set the temp
  #                 directory.
  monkeypatch_os_link()
  writer = AtomicWriter(path, **kwargs)
  return writer._open(_get_fileobject_func(writer, get_tmpdir_on_same_filesystem(path)))


def atomic_write_in_dir(path, **kwargs):
  """Creates an atomic writer using a temporary file in the same directory
     as the destination file.
  """
  monkeypatch_os_link()
  writer = AtomicWriter(path, **kwargs)
  return writer._open(_get_fileobject_func(writer, os.path.dirname(path)))
