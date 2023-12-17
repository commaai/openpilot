import os
import shutil
import tempfile
import contextlib
from typing import Optional


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

@contextlib.contextmanager
def atomic_write_on_fs_tmp(path: str, mode: str = 'w', buffering: int = -1, encoding: Optional[str] = None, newline: Optional[str] = None):
  """Write to a file atomically using a temporary file in a temporary directory on the same filesystem as path."""
  temp_dir = get_tmpdir_on_same_filesystem(path)
  with tempfile.NamedTemporaryFile(mode=mode, buffering=buffering, encoding=encoding, newline=newline, dir=temp_dir, delete=False) as tmp_file:
    yield tmp_file
    tmp_file_name = tmp_file.name
  os.replace(tmp_file_name, path)

@contextlib.contextmanager
def atomic_write_in_dir(path: str, mode: str = 'w', buffering: int = -1, encoding: Optional[str] = None, newline: Optional[str] = None, 
                        overwrite: bool = False):
  """Write to a file atomically using a temporary file in the same directory as the destination file."""
  dir_name = os.path.dirname(path)

  if not overwrite and os.path.exists(path):
    raise FileExistsError(f"File '{path}' already exists. To overwrite it, set 'overwrite' to True.")

  with tempfile.NamedTemporaryFile(mode=mode, buffering=buffering, encoding=encoding, newline=newline, dir=dir_name, delete=False) as tmp_file:
    yield tmp_file
    tmp_file_name = tmp_file.name
  os.replace(tmp_file_name, path)
