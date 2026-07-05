import os
import select
import subprocess
import json
import logging
import time
from collections.abc import Iterator
from collections import OrderedDict

import numpy as np
from openpilot.tools.lib.filereader import FileReader, resolve_name
from openpilot.tools.lib.exceptions import DataUnreadableError
from openpilot.tools.lib.vidindex import hevc_index

logger = logging.getLogger("tools")

HEVC_SLICE_B = 0
HEVC_SLICE_P = 1
HEVC_SLICE_I = 2

class LRUCache:
  def __init__(self, capacity: int):
    self._cache: OrderedDict = OrderedDict()
    self.capacity = capacity

  def __getitem__(self, key):
    self._cache.move_to_end(key)
    return self._cache[key]

  def __setitem__(self, key, value):
    self._cache[key] = value
    if len(self._cache) > self.capacity:
        self._cache.popitem(last=False)

  def __contains__(self, key):
    return key in self._cache

def assert_hvec(fn: str) -> None:
  with FileReader(fn) as f:
    header = f.read(4)
  if len(header) == 0:
    raise DataUnreadableError(f"{fn} is empty")
  elif header == b"\x00\x00\x00\x01":
    if 'hevc' not in fn:
      raise NotImplementedError(fn)

def decompress_video_data(rawdat, w, h, pix_fmt="rgb24", vid_fmt='hevc', hwaccel="auto", loglevel="info") -> np.ndarray:
  threads = os.getenv("FFMPEG_THREADS", "0")
  args = ["ffmpeg", "-v", loglevel,
          "-threads", threads,
          "-hwaccel", hwaccel,
          "-c:v", "hevc",
          "-vsync", "0",
          "-f", vid_fmt,
          "-flags2", "showall",
          "-i", "pipe:0",
          "-f", "rawvideo",
          "-pix_fmt", pix_fmt,
          "pipe:1"]
  dat = subprocess.check_output(args, input=rawdat)

  ret: np.ndarray
  if pix_fmt == "rgb24":
    ret = np.frombuffer(dat, dtype=np.uint8).reshape(-1, h, w, 3)
  elif pix_fmt in ["nv12", "yuv420p"]:
    ret = np.frombuffer(dat, dtype=np.uint8).reshape(-1, (h*w*3//2))
  else:
    raise NotImplementedError(f"Unsupported pixel format: {pix_fmt}")
  return ret

def frame_size(width: int, height: int, pix_fmt: str) -> int:
  if pix_fmt == "rgb24":
    return width * height * 3
  elif pix_fmt in ("nv12", "yuv420p"):
    return width * height * 3 // 2
  else:
    raise NotImplementedError(f"Unsupported pixel format: {pix_fmt}")

class FfmpegStreamDecoder:
  def __init__(self, w: int, h: int, pix_fmt: str = "nv12", vid_fmt: str = "hevc",
               hwaccel: str = "auto", loglevel: str = "quiet", read_timeout: float = 0.05,
               startup_timeout: float = 0.35):
    self.w = w
    self.h = h
    self.pix_fmt = pix_fmt
    self.frame_size = frame_size(w, h, pix_fmt)
    self.read_timeout = read_timeout
    self.startup_timeout = startup_timeout
    self.started = False
    self._stdout_buf = bytearray()
    self._frames: list[np.ndarray] = []
    self._stderr = bytearray()

    threads = os.getenv("FFMPEG_STREAM_THREADS", "1")
    args = ["ffmpeg", "-v", loglevel,
            "-threads", threads,
            "-hwaccel", hwaccel,
            "-probesize", "32768",
            "-analyzeduration", "0",
            "-f", vid_fmt,
            "-flags2", "showall",
            "-i", "pipe:0",
            "-f", "rawvideo",
            "-pix_fmt", pix_fmt,
            "pipe:1"]
    self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert self.proc.stdout is not None
    assert self.proc.stderr is not None
    os.set_blocking(self.proc.stdout.fileno(), False)
    os.set_blocking(self.proc.stderr.fileno(), False)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self) -> None:
    if self.proc.poll() is None:
      if self.proc.stdin:
        try:
          self.proc.stdin.close()
        except BrokenPipeError:
          pass
      self.proc.terminate()
      try:
        self.proc.wait(timeout=1)
      except subprocess.TimeoutExpired:
        self.proc.kill()
        self.proc.wait()

  def _drain_fd(self, pipe, dst: bytearray, timeout: float) -> None:
    fd = pipe.fileno()
    end_time = timeout + time.monotonic()
    while True:
      poll_timeout = max(0.0, min(0.005, end_time - time.monotonic()))
      ready, _, _ = select.select([fd], [], [], poll_timeout)
      if not ready:
        return

      try:
        chunk = os.read(fd, 1024 * 1024)
      except BlockingIOError:
        continue
      if len(chunk) == 0:
        return
      dst.extend(chunk)
      if time.monotonic() >= end_time:
        return

  def _drain(self, timeout: float | None = None) -> None:
    assert self.proc.stdout is not None
    assert self.proc.stderr is not None
    self._drain_fd(self.proc.stdout, self._stdout_buf, self.read_timeout if timeout is None else timeout)
    self._drain_fd(self.proc.stderr, self._stderr, 0)

    while len(self._stdout_buf) >= self.frame_size:
      frame = bytes(self._stdout_buf[:self.frame_size])
      del self._stdout_buf[:self.frame_size]
      self._frames.append(np.frombuffer(frame, dtype=np.uint8))

  def _check_proc(self) -> None:
    return_code = self.proc.poll()
    if return_code is None:
      return
    self._drain(0)
    stderr = self._stderr.decode("utf-8", "replace").strip()
    raise RuntimeError(f"ffmpeg stream decoder exited with code {return_code}: {stderr}")

  def decode(self, packet: bytes | bytearray | memoryview) -> np.ndarray | None:
    if self.proc.stdin is None:
      raise RuntimeError("ffmpeg stream decoder stdin is closed")

    self._check_proc()
    try:
      self.proc.stdin.write(packet)
      self.proc.stdin.flush()
    except BrokenPipeError as e:
      self._check_proc()
      raise RuntimeError("ffmpeg stream decoder pipe closed") from e
    self._drain(self.read_timeout if self.started else self.startup_timeout)
    self._check_proc()

    if not self._frames:
      return None
    self.started = True
    return self._frames.pop(0)

def ffprobe(fn, fmt=None):
  fn = resolve_name(fn)
  cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams"]
  if fmt:
    cmd += ["-f", fmt]
  cmd += ["-i", "pipe:0"]

  try:
    with FileReader(fn) as f:
      ffprobe_output = subprocess.check_output(cmd, input=f.read(4096))
  except subprocess.CalledProcessError as e:
    raise DataUnreadableError(fn) from e
  return json.loads(ffprobe_output)

def get_index_data(fn: str, index_data: dict|None = None):
  if index_data is None:
    index_data = get_video_index(fn)
    if index_data is None:
      raise DataUnreadableError(f"Failed to index {fn!r}")
  stream = index_data["probe"]["streams"][0]
  return index_data["index"], index_data["global_prefix"], stream["width"], stream["height"]

def get_video_index(fn):
  assert_hvec(fn)
  frame_types, dat_len, prefix = hevc_index(fn)
  index = np.array(frame_types + [(0xFFFFFFFF, dat_len)], dtype=np.uint32)
  probe = ffprobe(fn, "hevc")
  return {
    'index': index,
    'global_prefix': prefix,
    'probe': probe
  }

class FfmpegDecoder:
  def __init__(self, fn: str, index_data: dict|None = None,
               pix_fmt: str = "rgb24", hwaccel="auto", loglevel="quiet"):
    self.fn = fn
    self.index, self.prefix, self.w, self.h = get_index_data(fn, index_data)
    self.frame_count = len(self.index) - 1          # sentinel row at the end
    self.iframes = np.where(self.index[:, 0] == HEVC_SLICE_I)[0]
    self.pix_fmt = pix_fmt
    self.loglevel, self.hwaccel = loglevel, hwaccel

  def _gop_bounds(self, frame_idx: int):
    f_b = frame_idx
    while f_b > 0 and self.index[f_b, 0] != HEVC_SLICE_I:
      f_b -= 1
    f_e = frame_idx + 1
    while f_e < self.frame_count and self.index[f_e, 0] != HEVC_SLICE_I:
      f_e += 1
    return f_b, f_e, self.index[f_b, 1], self.index[f_e, 1]

  def _decode_gop(self, raw: bytes) -> Iterator[np.ndarray]:
    yield from decompress_video_data(raw, self.w, self.h, pix_fmt=self.pix_fmt, hwaccel=self.hwaccel, loglevel=self.loglevel)

  def get_gop_start(self, frame_idx: int):
    return self.iframes[np.searchsorted(self.iframes, frame_idx, side="right") - 1]

  def get_iterator(self, start_fidx: int = 0, end_fidx: int|None = None,
                   frame_skip: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    end_fidx = end_fidx or self.frame_count
    fidx = start_fidx
    while fidx < end_fidx:
      f_b, f_e, off_b, off_e = self._gop_bounds(fidx)
      with FileReader(self.fn) as f:
        f.seek(off_b)
        raw = self.prefix + f.read(off_e - off_b)
      # number of frames to discard inside this GOP before the wanted one
      for i, frm in enumerate(decompress_video_data(raw, self.w, self.h, self.pix_fmt, hwaccel=self.hwaccel, loglevel=self.loglevel)):
        fidx = f_b + i
        if fidx >= end_fidx:
          return
        elif fidx >= start_fidx and (fidx - start_fidx) % frame_skip == 0:
          yield fidx, frm
      fidx += 1

def FrameIterator(fn: str, index_data: dict|None=None, pix_fmt: str = "rgb24",
                  start_fidx:int=0, end_fidx=None, frame_skip:int=1, hwaccel="auto", loglevel="quiet") -> Iterator[np.ndarray]:
  dec = FfmpegDecoder(fn, pix_fmt=pix_fmt, index_data=index_data, hwaccel=hwaccel, loglevel=loglevel)
  for _, frame in dec.get_iterator(start_fidx=start_fidx, end_fidx=end_fidx, frame_skip=frame_skip):
    yield frame

class FrameReader:
  def __init__(self, fn: str, index_data: dict|None = None, cache_size: int = 30,
               pix_fmt: str = "rgb24", hwaccel="auto", loglevel="quiet"):
    self.decoder = FfmpegDecoder(fn, index_data=index_data, pix_fmt=pix_fmt, hwaccel=hwaccel, loglevel=loglevel)
    self.iframes = self.decoder.iframes
    self._cache: LRUCache = LRUCache(cache_size)
    self.w, self.h, self.frame_count, = self.decoder.w, self.decoder.h, self.decoder.frame_count
    self.pix_fmt = pix_fmt

    self.it: Iterator[tuple[int, np.ndarray]] | None = None
    self.fidx = -1

  def get(self, fidx:int):
    if fidx in self._cache:  # If frame is cached, return it
      return self._cache[fidx]
    read_start = self.decoder.get_gop_start(fidx)
    if not self.it or fidx < self.fidx or read_start != self.decoder.get_gop_start(self.fidx):  # If the frame is in a different GOP, reset the iterator
      self.it = self.decoder.get_iterator(read_start)
      self.fidx = -1
    while self.fidx < fidx:
      self.fidx, frame = next(self.it)
      self._cache[self.fidx] = frame
    return self._cache[fidx]
