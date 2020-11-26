import os
from atomicwrites import AtomicWriter


def atomic_write_in_dir(path, **kwargs):
  """Creates an atomic writer using a temporary file in the same directory
     as the destination file.
  """
  writer = AtomicWriter(path, **kwargs)
  return writer._open(_get_fileobject_func(writer, os.path.dirname(path)))


def _get_fileobject_func(writer, temp_dir):
  def _get_fileobject():
    file_obj = writer.get_fileobject(dir=temp_dir)
    os.chmod(file_obj.name, 0o644)
    return file_obj
  return _get_fileobject


def mkdirs_exists_ok(path):
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise
