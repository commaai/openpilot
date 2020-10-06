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
    return "/{}/runner/tmp".format(parts[1])
  return "/tmp"


class AutoMoveTempdir():
  def __init__(self, target_path, temp_dir=None):
    self._target_path = target_path
    self._path = tempfile.mkdtemp(dir=temp_dir)

  @property
  def name(self):
    return self._path

  def close(self):
    os.rename(self._path, self._target_path)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_type is None:
      self.close()
    else:
      shutil.rmtree(self._path)


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


def _get_fileobject_func(writer, temp_dir):
  def _get_fileobject():
    file_obj = writer.get_fileobject(dir=temp_dir)
    os.chmod(file_obj.name, 0o644)
    return file_obj
  return _get_fileobject


def atomic_write_on_fs_tmp(path, **kwargs):
  """Creates an atomic writer using a temporary file in a temporary directory
     on the same filesystem as path.
  """
  # TODO(mgraczyk): This use of AtomicWriter relies on implementation details to set the temp
  #                 directory.
  writer = AtomicWriter(path, **kwargs)
  return writer._open(_get_fileobject_func(writer, get_tmpdir_on_same_filesystem(path)))


def atomic_write_in_dir(path, **kwargs):
  """Creates an atomic writer using a temporary file in the same directory
     as the destination file.
  """
  writer = AtomicWriter(path, **kwargs)
  return writer._open(_get_fileobject_func(writer, os.path.dirname(path)))


def atomic_write_in_dir_neos(path, contents, mode=None):
  """
  Atomically writes contents to path using a temporary file in the same directory
  as path. Useful on NEOS, where `os.link` (required by atomic_write_in_dir) is missing.
  """

  f = tempfile.NamedTemporaryFile(delete=False, prefix=".tmp", dir=os.path.dirname(path))
  f.write(contents)
  f.flush()
  if mode is not None:
    os.fchmod(f.fileno(), mode)
  os.fsync(f.fileno())
  f.close()

  os.rename(f.name, path)
