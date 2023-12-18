import os
import tempfile
import contextlib
from typing import Optional


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
