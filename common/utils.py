import io
import os
import tempfile
import contextlib
import subprocess
import time
import functools
from subprocess import Popen, PIPE, TimeoutExpired
import zstandard as zstd

LOG_COMPRESSION_LEVEL = 10  # little benefit up to level 15. level ~17 is a small step change

class Timer:
  """Simple lap timer for profiling sequential operations."""

  def __init__(self):
    self._start = self._lap = time.monotonic()
    self._sections = {}

  def lap(self, name):
    now = time.monotonic()
    self._sections[name] = now - self._lap
    self._lap = now

  @property
  def total(self):
    return time.monotonic() - self._start

  def fmt(self, duration):
    parts = ", ".join(f"{k}={v:.2f}s" + (f" ({duration/v:.0f}x)" if k == 'render' and v > 0 else "") for k, v in self._sections.items())
    total = self.total
    realtime = f"{duration/total:.1f}x realtime" if total > 0 else "N/A"
    return f"{duration}s in {total:.1f}s ({realtime}) | {parts}"

def sudo_write(val: str, path: str) -> None:
  try:
    with open(path, 'w') as f:
      f.write(str(val))
  except PermissionError:
    os.system(f"sudo chmod a+w {path}")
    try:
      with open(path, 'w') as f:
        f.write(str(val))
    except PermissionError:
      # fallback for debugfs files
      os.system(f"sudo su -c 'echo {val} > {path}'")


def sudo_read(path: str) -> str:
  try:
    return subprocess.check_output(f"sudo cat {path}", shell=True, encoding='utf8').strip()
  except Exception:
    return ""


class MovingAverage:
  def __init__(self, window_size: int):
    self.window_size: int = window_size
    self.buffer: list[float] = [0.0] * window_size
    self.index: int = 0
    self.count: int = 0
    self.sum: float = 0.0

  def add_value(self, new_value: float):
    # Update the sum: subtract the value being replaced and add the new value
    self.sum -= self.buffer[self.index]
    self.buffer[self.index] = new_value
    self.sum += new_value

    # Update the index in a circular manner
    self.index = (self.index + 1) % self.window_size

    # Track the number of added values (for partial windows)
    self.count = min(self.count + 1, self.window_size)

  def get_average(self) -> float:
    if self.count == 0:
      return float('nan')
    return self.sum / self.count


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


def tabulate(tabular_data, headers=(), tablefmt="simple", floatfmt="g", stralign="left", numalign=None):
  rows = [list(row) for row in tabular_data]

  def fmt(val):
    if isinstance(val, str):
      return val
    if isinstance(val, (bool, int)):
      return str(val)
    try:
      return format(val, floatfmt)
    except (TypeError, ValueError):
      return str(val)

  formatted = [[fmt(c) for c in row] for row in rows]
  hdrs = [str(h) for h in headers] if headers else None

  ncols = max((len(r) for r in formatted), default=0)
  if hdrs:
    ncols = max(ncols, len(hdrs))
  if ncols == 0:
    return ""

  for r in formatted:
    r.extend([""] * (ncols - len(r)))
  if hdrs:
    hdrs.extend([""] * (ncols - len(hdrs)))

  widths = [0] * ncols
  if hdrs:
    for i in range(ncols):
      widths[i] = len(hdrs[i])
  for row in formatted:
    for i in range(ncols):
      widths[i] = max(widths[i], max(len(ln) for ln in row[i].split('\n')))

  def _align(s, w):
    if stralign == "center":
      return s.center(w)
    return s.ljust(w)

  if tablefmt == "html":
    parts = ["<table>"]
    if hdrs:
      parts.append("<thead>")
      parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in hdrs) + "</tr>")
      parts.append("</thead>")
    parts.append("<tbody>")
    for row in formatted:
      parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts)

  if tablefmt == "simple_grid":
    def _sep(left, mid, right):
      return left + mid.join("\u2500" * (w + 2) for w in widths) + right

    top, mid_sep, bot = _sep("\u250c", "\u252c", "\u2510"), _sep("\u251c", "\u253c", "\u2524"), _sep("\u2514", "\u2534", "\u2518")

    def _fmt_row(cells):
      split = [c.split('\n') for c in cells]
      nlines = max(len(s) for s in split)
      for s in split:
        s.extend([""] * (nlines - len(s)))
      return ["\u2502" + "\u2502".join(f" {_align(split[i][li], widths[i])} " for i in range(ncols)) + "\u2502" for li in range(nlines)]

    lines = [top]
    if hdrs:
      lines.extend(_fmt_row(hdrs))
      lines.append(mid_sep)
    for ri, row in enumerate(formatted):
      lines.extend(_fmt_row(row))
      lines.append(mid_sep if ri < len(formatted) - 1 else bot)
    return "\n".join(lines)

  # simple
  gap = "  "
  lines = []
  if hdrs:
    lines.append(gap.join(h.ljust(w) for h, w in zip(hdrs, widths, strict=True)))
    lines.append(gap.join("-" * w for w in widths))
  for row in formatted:
    lines.append(gap.join(_align(row[i], widths[i]) for i in range(ncols)))
  return "\n".join(lines)


def retry(attempts=3, delay=1.0, ignore_failure=False):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      for _ in range(attempts):
        try:
          return func(*args, **kwargs)
        except Exception:
          print(f"{func.__name__} failed, trying again")
          time.sleep(delay)

      if ignore_failure:
        print(f"{func.__name__} failed after retry")
      else:
        raise Exception(f"{func.__name__} failed after retry")
    return wrapper
  return decorator
