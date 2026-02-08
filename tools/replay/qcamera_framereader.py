#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
import threading

import numpy as np

from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.framereader import LRUCache


def _parse_fraction(rate: str | None) -> float:
  if not rate:
    return 0.0
  try:
    num, den = rate.split("/")
    den_v = float(den)
    return float(num) / den_v if den_v != 0 else 0.0
  except Exception:
    return 0.0


def _probe_video(fn: str) -> tuple[int, int, int]:
  cmd = [
    "ffprobe",
    "-v",
    "quiet",
    "-print_format",
    "json",
    "-show_format",
    "-show_streams",
    "-i",
    "-",
  ]

  with FileReader(fn) as f:
    header = f.read(2 * 1024 * 1024)
  probe = json.loads(subprocess.check_output(cmd, input=header))

  stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
  if stream is None:
    raise RuntimeError(f"No video stream in {fn}")

  width = int(stream.get("width") or 0)
  height = int(stream.get("height") or 0)
  if width <= 0 or height <= 0:
    raise RuntimeError(f"Invalid frame size for {fn}")

  frame_count = int(stream.get("nb_frames") or 0)
  if frame_count <= 0:
    fps = _parse_fraction(stream.get("avg_frame_rate"))
    duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    if fps > 0 and duration > 0:
      frame_count = int(max(1.0, round(fps * duration)))

  if frame_count <= 0:
    frame_count = 1200

  return width, height, frame_count


class QCameraFrameReader:
  def __init__(self, fn: str, cache_size: int = 30, pix_fmt: str = "nv12"):
    if pix_fmt != "nv12":
      raise NotImplementedError("QCameraFrameReader only supports nv12")

    self.fn = fn
    self.w, self.h, self.frame_count = _probe_video(fn)
    self._frame_size = self.w * self.h * 3 // 2
    self._cache: LRUCache = LRUCache(cache_size)
    self._decode_lock = threading.Lock()
    self._raw_path: str | None = None
    self._raw_file = None

  def __del__(self):
    if self._raw_file is not None:
      try:
        self._raw_file.close()
      except Exception:
        pass

    if self._raw_path is not None:
      try:
        os.unlink(self._raw_path)
      except OSError:
        pass

  def _ensure_decoded(self) -> None:
    if self._raw_file is not None:
      return

    with self._decode_lock:
      if self._raw_file is not None:
        return

      with tempfile.NamedTemporaryFile(prefix="replay_qcam_", suffix=".nv12", delete=False) as tmp:
        with FileReader(self.fn) as f:
          compressed = f.read()

        proc = subprocess.run(
          ["ffmpeg", "-v", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
          input=compressed,
          stdout=tmp,
          stderr=subprocess.PIPE,
          check=False,
        )
        out_path = tmp.name
        out_size = tmp.tell()

      if proc.returncode != 0:
        try:
          os.unlink(out_path)
        except OSError:
          pass
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg qcamera decode failed: {stderr}")

      self._raw_path = out_path
      self._raw_file = open(out_path, "rb")
      decoded_count = out_size // self._frame_size
      if decoded_count > 0:
        self.frame_count = decoded_count

  def get(self, fidx: int) -> np.ndarray:
    if fidx in self._cache:
      return self._cache[fidx]

    self._ensure_decoded()
    if self._raw_file is None:
      raise RuntimeError("qcamera decoder is not initialized")

    if fidx < 0 or fidx >= self.frame_count:
      raise IndexError(f"frame index {fidx} out of range 0..{self.frame_count - 1}")

    self._raw_file.seek(fidx * self._frame_size)
    data = self._raw_file.read(self._frame_size)
    if len(data) != self._frame_size:
      raise RuntimeError(f"short read while reading qcamera frame {fidx}")

    frame = np.frombuffer(data, dtype=np.uint8).copy()
    self._cache[fidx] = frame
    return frame
