#!/usr/bin/env python3
import logging
import queue
import threading
import time
from enum import Enum
from typing import Optional

import numpy as np

from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.tools.lib.framereader import FrameReader

log = logging.getLogger("replay")

BUFFER_COUNT = 40


def get_nv12_info(width: int, height: int) -> tuple[int, int, int]:
  """Calculate NV12 buffer parameters matching C++ VENUS macros."""
  # VENUS_Y_STRIDE for NV12: align to 128
  nv12_width = (width + 127) & ~127
  # VENUS_Y_SCANLINES for NV12: align to 32
  nv12_height = (height + 31) & ~31
  # Buffer size from v4l2_format (matches C++ implementation)
  nv12_buffer_size = 2346 * nv12_width
  return nv12_width, nv12_height, nv12_buffer_size


def repack_nv12_to_venus(yuv: np.ndarray, width: int, height: int, stride: int) -> np.ndarray:
  """Repack NV12 data from unpadded to VENUS-aligned stride.

  FrameReader returns NV12 with original width as stride.
  VisionIPC expects VENUS-aligned stride (128-byte aligned).
  """
  y_plane_size = width * height

  # Reshape into planes
  y_plane = yuv[:y_plane_size].reshape(height, width)
  uv_plane = yuv[y_plane_size:].reshape(height // 2, width)

  # Calculate VENUS-aligned scanlines
  y_scanlines = (height + 31) & ~31
  uv_scanlines = (height // 2 + 31) & ~31

  # Create padded planes with VENUS stride using numpy pad
  # Pad width: (0 padding before, stride-width padding after) for axis 1
  y_padded = np.pad(y_plane, ((0, y_scanlines - height), (0, stride - width)), mode='constant')
  uv_padded = np.pad(uv_plane, ((0, uv_scanlines - height // 2), (0, stride - width)), mode='constant')

  # Concatenate and flatten
  return np.concatenate([y_padded.ravel(), uv_padded.ravel()])


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
    self.nv12_buffer_size = 0  # Padded buffer size for VisionIPC
    self.thread: Optional[threading.Thread] = None
    self.queue: queue.Queue = queue.Queue()
    self.prefetch_up_to: int = -1  # Highest frame index we've prefetched


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
    self._vipc_server = VisionIpcServer("camerad")

    for cam in self._cameras.values():
      if cam.width > 0 and cam.height > 0:
        nv12_width, nv12_height, nv12_buffer_size = get_nv12_info(cam.width, cam.height)
        cam.nv12_stride = nv12_width
        cam.nv12_buffer_size = nv12_buffer_size
        log.info(f"camera[{cam.type.name}] frame size {cam.width}x{cam.height}, stride {nv12_width}, buffer {nv12_buffer_size}")
        self._vipc_server.create_buffers_with_sizes(
          cam.stream_type, BUFFER_COUNT, cam.width, cam.height,
          nv12_buffer_size, nv12_width, nv12_width * nv12_height
        )

        if cam.thread is None or not cam.thread.is_alive():
          cam.thread = threading.Thread(
            target=self._camera_thread,
            args=(cam,),
            daemon=True
          )
          cam.thread.start()

    self._vipc_server.start_listener()

  def _camera_thread(self, cam: Camera) -> None:
    current_fr: Optional[FrameReader] = None
    prefetch_ahead = 60  # Stay 2 GOPs ahead

    while not self._exit:
      # Try to get next frame request, but don't block long - we want to prefetch
      try:
        item = cam.queue.get(timeout=0.005)  # 5ms timeout for responsive prefetching
      except queue.Empty:
        # No frame requested - use idle time to prefetch
        if current_fr is not None and cam.prefetch_up_to < current_fr.frame_count - 1:
          # Prefetch next frame sequentially
          cam.prefetch_up_to += 1
          self._get_frame(current_fr, cam.prefetch_up_to)
        continue

      if item is None:  # Termination signal
        break

      fr, event = item
      current_fr = fr

      try:
        # Get encode index from the event
        eidx = event.roadEncodeIdx if cam.type == CameraType.ROAD else \
               event.driverEncodeIdx if cam.type == CameraType.DRIVER else \
               event.wideRoadEncodeIdx

        local_frame_idx = eidx.segmentId  # segmentId is actually the local frame index within segment
        frame_id = eidx.frameId

        # Update prefetch target if we've caught up
        if cam.prefetch_up_to < local_frame_idx:
          cam.prefetch_up_to = local_frame_idx

        # Get the frame (should be cached from prefetch)
        yuv = self._get_frame(fr, local_frame_idx)
        if yuv is not None:
          # Repack from unpadded NV12 to VENUS-aligned stride
          yuv_venus = repack_nv12_to_venus(yuv, cam.width, cam.height, cam.nv12_stride)
          yuv_bytes = yuv_venus.tobytes()
          # Pad to match the full buffer size expected by VisionIPC
          if len(yuv_bytes) < cam.nv12_buffer_size:
            yuv_bytes = yuv_bytes + bytes(cam.nv12_buffer_size - len(yuv_bytes))

          timestamp_sof = eidx.timestampSof
          timestamp_eof = eidx.timestampEof
          self._vipc_server.send(cam.stream_type, yuv_bytes, frame_id, timestamp_sof, timestamp_eof)

        # Aggressively prefetch if we're not far enough ahead
        # This ensures we decode the next GOP before we need it
        target = min(local_frame_idx + prefetch_ahead, fr.frame_count - 1)
        while cam.prefetch_up_to < target and cam.queue.empty():
          cam.prefetch_up_to += 1
          self._get_frame(fr, cam.prefetch_up_to)

      except Exception:
        log.exception(f"camera[{cam.type.name}] error")

      with self._publishing_lock:
        self._publishing -= 1

  def _get_frame(self, fr: FrameReader, local_idx: int) -> Optional[np.ndarray]:
    """Get frame from FrameReader. FrameReader has its own LRU cache."""
    try:
      if local_idx < fr.frame_count:
        return fr.get(local_idx)
    except Exception as e:
      log.warning(f"Failed to decode frame {local_idx}: {e}")
    return None

  def push_frame(self, cam_type: CameraType, fr: FrameReader, event) -> None:
    cam = self._cameras[cam_type]

    # Check if frame size changed
    if cam.width != fr.w or cam.height != fr.h:
      cam.width = fr.w
      cam.height = fr.h
      self.wait_for_sent()
      self._start_vipc_server()

    with self._publishing_lock:
      self._publishing += 1
    cam.queue.put((fr, event))

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
    log.info(f"warming cache: frames {start_frame}-{end_frame}")
    for i in range(start_frame, end_frame):
      self._get_frame(fr, i)
