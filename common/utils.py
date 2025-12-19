import io
import os
import tempfile
import contextlib
import subprocess
import time
import functools
from subprocess import Popen, PIPE, TimeoutExpired
import zstandard as zstd
from openpilot.common.swaglog import cloudlog

LOG_COMPRESSION_LEVEL = 10  # little benefit up to level 15. level ~17 is a small step change


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
def atomic_write(path: str, mode: str = 'w', buffering: int = -1, encoding: str | None = None, newline: str | None = None,
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


# remove all keys that end in DEPRECATED
def strip_deprecated_keys(d):
  for k in list(d.keys()):
    if isinstance(k, str):
      if k.endswith('DEPRECATED'):
        d.pop(k)
      elif isinstance(d[k], dict):
        strip_deprecated_keys(d[k])
  return d


def run_cmd(cmd: list[str], cwd=None, env=None) -> str:
  return subprocess.check_output(cmd, encoding='utf8', cwd=cwd, env=env).strip()


def run_cmd_default(cmd: list[str], default: str = "", cwd=None, env=None) -> str:
  try:
    return run_cmd(cmd, cwd=cwd, env=env)
  except subprocess.CalledProcessError:
    return default


@contextlib.contextmanager
def managed_proc(cmd: list[str], env: dict[str, str]):
  proc = Popen(cmd, env=env, stdout=PIPE, stderr=PIPE)
  try:
    yield proc
  finally:
    if proc.poll() is None:
      proc.terminate()
    try:
      proc.wait(timeout=5)
    except TimeoutExpired:
      proc.kill()


def retry(attempts=3, delay=1.0, ignore_failure=False):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      for _ in range(attempts):
        try:
          return func(*args, **kwargs)
        except Exception:
          cloudlog.exception(f"{func.__name__} failed, trying again")
          time.sleep(delay)

      if ignore_failure:
        cloudlog.error(f"{func.__name__} failed after retry")
      else:
        raise Exception(f"{func.__name__} failed after retry")
    return wrapper
  return decorator
