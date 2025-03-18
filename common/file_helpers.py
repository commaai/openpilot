import io
import os
import tempfile
import contextlib
import zstandard as zstd

LOG_COMPRESSION_LEVEL = 10 # little benefit up to level 15. level ~17 is a small step change


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
def atomic_write_in_dir(path: str, mode: str = 'w', buffering: int = -1, encoding: str = None, newline: str = None,
                        overwrite: bool = False):
  """Write to a file atomically using a temporary file in the same directory as the destination file."""
  dir_name = os.path.dirname(path)

  if not overwrite and os.path.exists(path):
    raise FileExistsError(f"File '{path}' already exists. To overwrite it, set 'overwrite' to True.")

  with tempfile.NamedTemporaryFile(mode=mode, buffering=buffering, encoding=encoding, newline=newline, dir=dir_name, delete=False) as tmp_file:
    yield tmp_file
    tmp_file_name = tmp_file.name
  os.replace(tmp_file_name, path)


def get_upload_stream(filepath: str, should_compress: bool) -> tuple[io.BufferedIOBase, int]:
  if not should_compress:
    file_size = os.path.getsize(filepath)
    file_stream = open(filepath, "rb")
    return file_stream, file_size

  # Compress the file on the fly
  compressed_stream = io.BytesIO()
  compressor = zstd.ZstdCompressor(level=LOG_COMPRESSION_LEVEL)

  with open(filepath, "rb") as f:
    compressor.copy_stream(f, compressed_stream)
    compressed_size = compressed_stream.tell()
    compressed_stream.seek(0)
    return compressed_stream, compressed_size
