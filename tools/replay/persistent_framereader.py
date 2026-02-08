#!/usr/bin/env python3
import json
import logging
import os
import subprocess
import threading
from collections import OrderedDict

import numpy as np

from openpilot.tools.lib.framereader import FrameReader

log = logging.getLogger("replay")


def _parse_fraction(rate: str | None) -> float:
  if not rate:
    return 0.0
  try:
    num, den = rate.split("/")
    den_f = float(den)
    return float(num) / den_f if den_f > 0 else 0.0
  except Exception:
    return 0.0


def _probe_video(path: str) -> tuple[int, int, int]:
  cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path]
  probe = json.loads(subprocess.check_output(cmd))

  stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
  if stream is None:
    raise RuntimeError(f"No video stream in {path}")

  width = int(stream.get("width") or 0)
  height = int(stream.get("height") or 0)
  if width <= 0 or height <= 0:
    raise RuntimeError(f"Invalid video dimensions in {path}")

  frame_count = int(stream.get("nb_frames") or 0)
  if frame_count <= 0:
    fps = _parse_fraction(stream.get("avg_frame_rate"))
    duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    if fps > 0 and duration > 0:
      frame_count = max(1, int(round(fps * duration)))

  return width, height, frame_count


class PersistentFFmpegFrameReader:
  """Frame reader that keeps a single ffmpeg decoder process alive.

  This avoids spawning a new ffmpeg process per GOP (the default FrameReader path),
  which is a major CPU overhead during replay.
  """

  def __init__(self, fn: str, cache_size: int = 30, pix_fmt: str = "nv12", expected_frame_count: int | None = None):
    if pix_fmt != "nv12":
      raise NotImplementedError("PersistentFFmpegFrameReader currently supports only nv12")

    self.fn = fn
    self.pix_fmt = pix_fmt
    self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
    self._cache_capacity = cache_size
    self._lock = threading.Lock()
    self._cv = threading.Condition(self._lock)

    self._proc: subprocess.Popen | None = None
    self._decoder_thread: threading.Thread | None = None
    self._decode_idx = 0
    self._decode_done = False
    self._target_idx = -1
    self._stop = False
    self._fallback_reader: FrameReader | None = None
    self._prefetch_ahead = max(1, int(os.getenv("REPLAY_PERSISTENT_PREFETCH_AHEAD", "60")))

    self.w, self.h, probed_count = _probe_video(fn)
    self._frame_size = self.w * self.h * 3 // 2
    self.frame_count = expected_frame_count if expected_frame_count is not None else (probed_count or 1200)

  def __del__(self):
    self.close()

  def close(self) -> None:
    with self._cv:
      self._stop = True
      proc = self._proc
      self._proc = None
      thread = self._decoder_thread
      self._decoder_thread = None
      self._cv.notify_all()

    if proc is not None and proc.poll() is None:
      proc.terminate()
      try:
        proc.wait(timeout=1.0)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)

    if thread is not None:
      thread.join(timeout=1.0)

    if proc is None:
      return

    try:
      if proc.stdout is not None:
        proc.stdout.close()
    except Exception:
      pass

    try:
      if proc.stderr is not None:
        proc.stderr.close()
    except Exception:
      pass

  def _start_decoder_locked(self) -> None:
    if self._proc is not None:
      return

    threads = os.getenv("FFMPEG_THREADS", "0")
    hwaccel = os.getenv("FFMPEG_HWACCEL", "auto")
    cmd = [
      "ffmpeg",
      "-v",
      "error",
      "-threads",
      threads,
      "-hwaccel",
      hwaccel,
      "-vsync",
      "0",
      "-f",
      "hevc",
      "-i",
      self.fn,
      "-f",
      "rawvideo",
      "-pix_fmt",
      self.pix_fmt,
      "-",
    ]
    self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if self._decoder_thread is None or not self._decoder_thread.is_alive():
      self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
      self._decoder_thread.start()

  def _decoder_loop(self) -> None:
    while True:
      with self._cv:
        self._cv.wait_for(lambda: self._stop or (not self._decode_done and self._decode_idx <= self._target_idx))
        if self._stop:
          return

        frame = self._read_next_frame_locked()
        if frame is None:
          self._decode_done = True
          self.frame_count = min(self.frame_count, self._decode_idx)
          self._cv.notify_all()
          continue

        self._cache[self._decode_idx] = frame
        while len(self._cache) > self._cache_capacity:
          self._cache.popitem(last=False)
        self._decode_idx += 1
        self._cv.notify_all()

  def _read_next_frame_locked(self) -> np.ndarray | None:
    proc = self._proc
    if proc is None or proc.stdout is None:
      return None

    buf = bytearray(self._frame_size)
    view = memoryview(buf)
    read = 0
    while read < self._frame_size:
      chunk = proc.stdout.read(self._frame_size - read)
      if not chunk:
        return None
      n = len(chunk)
      view[read : read + n] = chunk
      read += n

    return np.frombuffer(buf, dtype=np.uint8)

  def _fallback_get(self, fidx: int) -> np.ndarray:
    if self._fallback_reader is None:
      log.debug("persistent decoder fallback to GOP reader at frame %d", fidx)
      self._fallback_reader = FrameReader(self.fn, pix_fmt=self.pix_fmt, cache_size=self._cache_capacity)
    return self._fallback_reader.get(fidx)

  def prefetch_to(self, fidx: int) -> None:
    if fidx < 0:
      return

    with self._cv:
      if self._stop or self._decode_done:
        return
      self._start_decoder_locked()
      if fidx > self._target_idx:
        self._target_idx = fidx
        self._cv.notify_all()

  def get(self, fidx: int) -> np.ndarray:
    if fidx < 0:
      raise IndexError(f"negative frame index {fidx}")

    if self.frame_count > 0 and fidx >= self.frame_count:
      raise IndexError(f"frame index {fidx} out of range 0..{self.frame_count - 1}")

    if fidx in self._cache:
      return self._cache[fidx]

    fallback = False
    with self._cv:
      if fidx in self._cache:
        return self._cache[fidx]

      # If requested frame is older than our forward decoder position and was evicted,
      # use the old GOP-based reader as a correctness fallback.
      if fidx < self._decode_idx:
        fallback = True
      else:
        self._start_decoder_locked()
        target = min(self.frame_count - 1, fidx + self._prefetch_ahead)
        if target > self._target_idx:
          self._target_idx = target
          self._cv.notify_all()

        while fidx not in self._cache and not self._decode_done and not self._stop:
          self._cv.wait(timeout=0.25)

        if fidx in self._cache:
          return self._cache[fidx]

        if self._decode_done:
          raise IndexError(f"frame index {fidx} beyond decoded stream ({self._decode_idx} frames)")

    if fallback:
      frame = self._fallback_get(fidx)
      with self._cv:
        self._cache[fidx] = frame
        while len(self._cache) > self._cache_capacity:
          self._cache.popitem(last=False)
      return frame

    raise RuntimeError(f"failed to decode frame {fidx}")
