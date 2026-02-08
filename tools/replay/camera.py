#!/usr/bin/env python3
import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.lib.framereader import FrameReader

log = logging.getLogger("replay")

BUFFER_COUNT = 40


def repack_nv12_to_venus(yuv: np.ndarray, width: int, height: int, stride: int, y_scanlines: int, uv_scanlines: int, out: np.ndarray) -> None:
  y_plane_size = width * height

  y_src = yuv[:y_plane_size].reshape(height, width)
  uv_src = yuv[y_plane_size:].reshape(height // 2, width)

  y_bytes = stride * y_scanlines
  y_dst = out[:y_bytes].reshape(y_scanlines, stride)
  uv_dst = out[y_bytes : y_bytes + stride * uv_scanlines].reshape(uv_scanlines, stride)

  y_dst[:height, :width] = y_src
  uv_dst[: height // 2, :width] = uv_src


class CameraType(Enum):
  ROAD = 0
  DRIVER = 1
  WIDE_ROAD = 2


CAMERA_STREAM_TYPES = {
  CameraType.ROAD: VisionStreamType.VISION_STREAM_ROAD,
  CameraType.DRIVER: VisionStreamType.VISION_STREAM_DRIVER,
  CameraType.WIDE_ROAD: VisionStreamType.VISION_STREAM_WIDE_ROAD,
}


class Camera:
  def __init__(self, cam_type: CameraType):
    self.type = cam_type
    self.stream_type = CAMERA_STREAM_TYPES[cam_type]
    self.width = 0
    self.height = 0
    self.nv12_stride = 0  # VENUS-aligned stride
    self.y_scanlines = 0
    self.uv_scanlines = 0
    self.nv12_buffer_size = 0  # Padded buffer size for VisionIPC
    self.repack_buffer: Optional[np.ndarray] = None
    self.thread: Optional[threading.Thread] = None
    self.queue: queue.Queue = queue.Queue()
    self.prefetch_up_to: int = -1  # Highest frame index we've prefetched


@dataclass(slots=True)
class FrameRequest:
  local_frame_idx: int
  frame_id: int
  timestamp_sof: int
  timestamp_eof: int


class CameraServer:
  def __init__(self, camera_sizes: Optional[dict[CameraType, tuple[int, int]]] = None):
    self._cameras = {
      CameraType.ROAD: Camera(CameraType.ROAD),
      CameraType.DRIVER: Camera(CameraType.DRIVER),
      CameraType.WIDE_ROAD: Camera(CameraType.WIDE_ROAD),
    }

    if camera_sizes:
      for cam_type, (w, h) in camera_sizes.items():
        self._cameras[cam_type].width = w
        self._cameras[cam_type].height = h

    self._vipc_server: Optional[VisionIpcServer] = None
    self._publishing = 0
    self._publishing_lock = threading.Lock()
    self._exit = False

    self._start_vipc_server()

  def __del__(self):
    self._exit = True
    for cam in self._cameras.values():
      if cam.thread is not None and cam.thread.is_alive():
        # Signal termination
        cam.queue.put(None)
        cam.thread.join()

  def _start_vipc_server(self) -> None:
    server = VisionIpcServer("camerad")
    self._vipc_server = server

    for cam in self._cameras.values():
      if cam.width > 0 and cam.height > 0:
        nv12_width, nv12_height, uv_scanlines, nv12_buffer_size = get_nv12_info(cam.width, cam.height)
        cam.nv12_stride = nv12_width
        cam.y_scanlines = nv12_height
        cam.uv_scanlines = uv_scanlines
        cam.nv12_buffer_size = nv12_buffer_size
        cam.repack_buffer = np.zeros(nv12_buffer_size, dtype=np.uint8)
        log.info(f"camera[{cam.type.name}] frame size {cam.width}x{cam.height}, stride {nv12_width}, buffer {nv12_buffer_size}")
        server.create_buffers_with_sizes(cam.stream_type, BUFFER_COUNT, cam.width, cam.height, nv12_buffer_size, nv12_width, nv12_width * nv12_height)

        if cam.thread is None or not cam.thread.is_alive():
          cam.thread = threading.Thread(target=self._camera_thread, args=(cam,), daemon=True)
          cam.thread.start()

    server.start_listener()

  def _camera_thread(self, cam: Camera) -> None:
    current_fr: Optional[Any] = None
    prefetch_ahead = 60  # Stay 2 GOPs ahead
    prefetch_budget = 4
    backlog_drop_threshold = 240
    backlog_keep = 120

    while not self._exit:
      # Try to get next frame request, but don't block long - we want to prefetch
      try:
        item = cam.queue.get(timeout=0.005)  # 5ms timeout for responsive prefetching
      except queue.Empty:
        # No frame requested - use idle time to prefetch
        if current_fr is not None and cam.prefetch_up_to < current_fr.frame_count - 1:
          if hasattr(current_fr, "prefetch_to"):
            target = min(cam.prefetch_up_to + prefetch_ahead, current_fr.frame_count - 1)
            current_fr.prefetch_to(target)
          else:
            # Prefetch next frame sequentially
            cam.prefetch_up_to += 1
            self._get_frame(current_fr, cam.prefetch_up_to)
        continue

      if item is None:  # Termination signal
        break

      # If producer is outrunning decode, coalesce queued requests and keep latest.
      if cam.queue.qsize() > backlog_drop_threshold:
        latest = item
        dropped = 0
        while cam.queue.qsize() > backlog_keep:
          try:
            nxt = cam.queue.get_nowait()
          except queue.Empty:
            break
          if nxt is None:
            item = None
            break
          latest = nxt
          dropped += 1

        if dropped:
          with self._publishing_lock:
            self._publishing = max(0, self._publishing - dropped)

        if item is None:
          break
        item = latest

      fr, req = item
      current_fr = fr

      try:
        local_frame_idx = req.local_frame_idx
        frame_id = req.frame_id

        # Update prefetch target if we've caught up
        if cam.prefetch_up_to < local_frame_idx:
          cam.prefetch_up_to = local_frame_idx

        # Get the frame (should be cached from prefetch)
        yuv = self._get_frame(fr, local_frame_idx)
        if yuv is not None:
          if cam.repack_buffer is None:
            cam.repack_buffer = np.zeros(cam.nv12_buffer_size, dtype=np.uint8)

          repack_nv12_to_venus(
            yuv,
            cam.width,
            cam.height,
            cam.nv12_stride,
            cam.y_scanlines,
            cam.uv_scanlines,
            cam.repack_buffer,
          )

          server = self._vipc_server
          if server is not None:
            server.send(cam.stream_type, cam.repack_buffer, frame_id, req.timestamp_sof, req.timestamp_eof)

        # Aggressively prefetch if we're not far enough ahead
        # This ensures we decode the next GOP before we need it
        target = min(local_frame_idx + prefetch_ahead, fr.frame_count - 1)
        if hasattr(fr, "prefetch_to"):
          fr.prefetch_to(target)
        else:
          n = 0
          while cam.prefetch_up_to < target and n < prefetch_budget:
            cam.prefetch_up_to += 1
            self._get_frame(fr, cam.prefetch_up_to)
            n += 1

      except Exception:
        log.exception(f"camera[{cam.type.name}] error")

      with self._publishing_lock:
        self._publishing -= 1

  def _get_frame(self, fr: Any, local_idx: int) -> Optional[np.ndarray]:
    """Get frame from FrameReader. FrameReader has its own LRU cache."""
    try:
      if local_idx < fr.frame_count:
        return fr.get(local_idx)
    except Exception as e:
      log.warning(f"Failed to decode frame {local_idx}: {e}")
    return None

  def push_frame(self, cam_type: CameraType, fr: Any, local_frame_idx: int, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> None:
    cam = self._cameras[cam_type]

    # Check if frame size changed
    if cam.width != fr.w or cam.height != fr.h:
      cam.width = fr.w
      cam.height = fr.h
      self.wait_for_sent()
      self._start_vipc_server()

    with self._publishing_lock:
      self._publishing += 1
    cam.queue.put(
      (
        fr,
        FrameRequest(
          local_frame_idx=local_frame_idx,
          frame_id=frame_id,
          timestamp_sof=timestamp_sof,
          timestamp_eof=timestamp_eof,
        ),
      )
    )

  def wait_for_sent(self) -> None:
    while True:
      with self._publishing_lock:
        if self._publishing <= 0:
          break
      time.sleep(0.001)

  def warm_cache(self, fr: FrameReader, start_frame: int = 0, num_gops: int = 3) -> None:
    """Pre-decode frames to warm the cache before playback starts.

    This prevents stutter at the start of playback by ensuring the first
    few GOPs are already decoded and cached.
    """
    gop_size = 30
    end_frame = min(start_frame + num_gops * gop_size, fr.frame_count)
    if hasattr(fr, "prefetch_to"):
      getattr(fr, "prefetch_to")(max(start_frame, end_frame - 1))
      return

    log.info(f"warming cache: frames {start_frame}-{end_frame}")
    for i in range(start_frame, end_frame):
      self._get_frame(fr, i)
