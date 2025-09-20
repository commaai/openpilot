#!/usr/bin/env python3
"""
Python rewrite of system/loggerd/encoderd.cc

Behavioral parity goals:
- One worker thread per available camera stream.
- Cross-encoder startup sync using a shared start_frame_id.
- Segment rotation every SEGMENT_LENGTH * MAIN_FPS frames since start_frame_id.
- Periodic thumbnail publish.

Encoding backends:
- PC: FFmpeg via PyAV (library `av`).
- QCOM: Direct V4L2 M2M via ioctl/mmap (implemented in Python).

Note: Actual encoder backends are stubbed here and implemented in follow-up steps.
"""
import argparse
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable

import cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.system.hardware.hw import PC
from openpilot.system.loggerd.encoder.ffmpeg_encoder import FfmpegEncoder
from openpilot.system.loggerd.encoder.v4l2_encoder import V4LEncoder
from openpilot.system.loggerd.encoder.jpeg_encoder import JpegEncoder
from types import SimpleNamespace

# Reuse constants and camera info structure from C++ header translated here

MAIN_FPS = 20

# Allow test override like in loggerd.h
SEGMENT_LENGTH = int(os.getenv("LOGGERD_SEGMENT_LENGTH", os.getenv("SEGMENT_LENGTH", "60")))


# --------------------------- Encoder settings & info ---------------------------

@dataclass
class EncoderSettings:
  encode_type: str  # mirror cereal::EncodeIndex::Type as string tag
  bitrate: int
  gop_size: int
  b_frames: int = 0

  @staticmethod
  def main_encoder_settings(in_width: int) -> "EncoderSettings":
    if in_width <= 1344:
      etype = "BIG_BOX_LOSSLESS" if PC() else "FULL_H_E_V_C"
      return EncoderSettings(etype, 5_000_000, 20)
    else:
      etype = "BIG_BOX_LOSSLESS" if PC() else "FULL_H_E_V_C"
      return EncoderSettings(etype, 10_000_000, 30)

  @staticmethod
  def qcam_encoder_settings() -> "EncoderSettings":
    return EncoderSettings("QCAMERA_H264", 256_000, 15)

  @staticmethod
  def stream_encoder_settings() -> "EncoderSettings":
    return EncoderSettings("QCAMERA_H264", 1_000_000, 15)


@dataclass
class EncoderInfo:
  publish_name: str
  filename: str | None = None
  thumbnail_name: str | None = None
  record: bool = True
  include_audio: bool = False
  frame_width: int = -1
  frame_height: int = -1
  fps: int = MAIN_FPS
  get_settings: Callable[[int], EncoderSettings] = field(default=lambda _: EncoderSettings.main_encoder_settings(1920))


@dataclass
class LogCameraInfo:
  thread_name: str
  stream_type: VisionStreamType
  fps: int = MAIN_FPS
  encoder_infos: list[EncoderInfo] = field(default_factory=list)


# --------------------------- Encoderd state & sync ----------------------------

class EncoderdState:
  def __init__(self) -> None:
    self.max_waiting: int = 0
    self.encoders_ready: int = 0
    self.start_frame_id: int = 0
    # index by VisionStreamType value
    self.camera_ready: dict[VisionStreamType, bool] = {}
    self.camera_synced: dict[VisionStreamType, bool] = {}
    self._lock = threading.Lock()

  def update_max_atomic(self, fid: int) -> None:
    with self._lock:
      if fid > self.start_frame_id:
        self.start_frame_id = fid


def sync_encoders(state: EncoderdState, cam_type: VisionStreamType, frame_id: int) -> bool:
  if state.camera_synced.get(cam_type, False):
    return True

  if state.max_waiting > 1 and state.encoders_ready != state.max_waiting:
    # add a small margin to the start frame id in case one of the encoders already dropped the next frame
    state.update_max_atomic(frame_id + 2)
    if not state.camera_ready.get(cam_type, False):
      state.camera_ready[cam_type] = True
      with state._lock:
        state.encoders_ready += 1
    return False
  else:
    if state.max_waiting == 1:
      state.update_max_atomic(frame_id)
    synced = frame_id >= state.start_frame_id
    state.camera_synced[cam_type] = synced
    return synced


# ------------------------------ Encoder backends ------------------------------

class BaseEncoder:
  def __init__(self, encoder_info: EncoderInfo, in_width: int, in_height: int):
    self.encoder_info = encoder_info
    self.in_width = in_width
    self.in_height = in_height
    self.out_width = encoder_info.frame_width if encoder_info.frame_width > 0 else in_width
    self.out_height = encoder_info.frame_height if encoder_info.frame_height > 0 else in_height
    self.pm = messaging.PubMaster([encoder_info.publish_name])
    self.encode_cnt = 0

  def open(self) -> None:
    raise NotImplementedError

  def close(self) -> None:
    raise NotImplementedError

  def encode_frame(self, buf: VisionBuf, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> int:
    """Return output packet index or -1 on failure."""
    raise NotImplementedError


class AvEncoder(BaseEncoder):
  """FFmpeg encoder using PyAV (PC path). Implementation to be filled in."""
  def open(self) -> None:
    import av  # lazy import
    settings = self.encoder_info.get_settings(self.in_width)
    codec_name = "h264" if settings.encode_type in ("QCAMERA_H264", "LIVESTREAM_H264") else "hevc"
    self._av = av
    self.codec = av.CodecContext.create(codec_name, "w")
    self.codec.width = self.out_width
    self.codec.height = self.out_height
    self.codec.time_base = av.time_base.TimeBase(1, self.encoder_info.fps)
    self.codec.framerate = av.time_base.TimeBase(self.encoder_info.fps, 1)
    self.codec.bit_rate = settings.bitrate
    self.codec.options = {"g": str(settings.gop_size), "bf": "0"}
    # Open encoder
    self.codec.open()
    self.segment_num = getattr(self, "segment_num", -1) + 1
    self.counter = 0

  def close(self) -> None:
    if hasattr(self, "codec") and self.codec:
      try:
        for pkt in self.codec.encode(None):
          # drop flush packets
          _ = pkt
      except Exception:
        pass
      self.codec = None

  def _publish_packet(self, pkt, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> None:
    # Build EncodeData message identical to C++ publisher_publish
    msg = messaging.new_message(self.encoder_info.publish_name)
    ed = getattr(msg, self.encoder_info.publish_name)
    ed.unixTimestampNanos = int(time.time_ns())
    idx = ed.idx
    idx.frameId = int(frame_id)
    idx.timestampSof = int(timestamp_sof)
    idx.timestampEof = int(timestamp_eof)
    # type optional; map known types
    # idx.type left default if mapping differs
    idx.encodeId = self.encode_cnt
    idx.segmentNum = int(self.segment_num)
    idx.segmentId = int(self.counter)
    flags = 8 if getattr(pkt, "is_keyframe", False) else 0  # V4L2_BUF_FLAG_KEYFRAME
    idx.flags = flags
    data = bytes(pkt)
    idx.len = len(data)
    ed.data = data
    ed.width = self.out_width
    ed.height = self.out_height
    # header: include codec extradata if present and keyframe
    if flags & 8:
      try:
        if self.codec and self.codec.extradata:
          ed.header = bytes(self.codec.extradata)
      except Exception:
        pass
    self.pm.send(self.encoder_info.publish_name, msg)
    self.encode_cnt += 1
    self.counter += 1

  def encode_frame(self, buf: VisionBuf, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> int:
    import numpy as np
    av = self._av
    # Extract Y and UV planes into contiguous arrays
    H, W, S = buf.height, buf.width, buf.stride
    y = np.asarray(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, S))[:H, :W].copy(order="C")
    uv_rows = H // 2
    uv = np.asarray(buf.data[buf.uv_offset:buf.uv_offset + uv_rows * S], dtype=np.uint8).reshape((uv_rows, S))[:, :W].copy(order="C")

    frame = av.VideoFrame(W, H, "nv12")
    frame.planes[0].update(y)
    frame.planes[1].update(uv)

    # Convert to encoder's pixel format if needed
    if self.codec.pix_fmt.name != "nv12":
      frame = frame.reformat(format=self.codec.pix_fmt.name, width=self.out_width, height=self.out_height)

    ret = 0
    for pkt in self.codec.encode(frame):
      self._publish_packet(pkt, frame_id, timestamp_sof, timestamp_eof)
      ret = self.counter
    return ret


class V4L2Encoder(BaseEncoder):
  """QCOM V4L2 M2M encoder (direct ioctl/mmap). 1:1 pipeline with dequeue thread."""
  def open(self) -> None:
    import os
    import fcntl
    import ctypes as C
    import select
    import mmap
    from openpilot.system.loggerd.v4l2_ctypes import (
      VIDIOC_QUERYCAP, VIDIOC_S_FMT, VIDIOC_S_PARM, VIDIOC_REQBUFS, VIDIOC_QUERYBUF,
      VIDIOC_STREAMON, VIDIOC_DQBUF, VIDIOC_QBUF,
      V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
      V4L2_MEMORY_MMAP, V4L2_MEMORY_USERPTR,
      V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_H264, V4L2_PIX_FMT_HEVC,
      V4L2_BUF_FLAG_KEYFRAME,
      V4L2_QCOM_BUF_FLAG_CODECCONFIG, V4L2_QCOM_BUF_FLAG_EOS,
      v4l2_capability, v4l2_format, v4l2_pix_format_mplane, v4l2_format_union,
      v4l2_streamparm, v4l2_streamparm_parm, v4l2_outputparm, v4l2_fract,
      v4l2_requestbuffers, v4l2_buffer, v4l2_plane, v4l2_buffer_m,
    )

    dev = "/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1"
    self._fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)

    cap = v4l2_capability()
    fcntl.ioctl(self._fd, VIDIOC_QUERYCAP, cap)

    # codec selection
    settings = self.encoder_info.get_settings(self.in_width)
    is_h265 = settings.encode_type in ("FULL_H_E_V_C",)
    out_pix = V4L2_PIX_FMT_HEVC if is_h265 else V4L2_PIX_FMT_H264

    # capture (encoded) format
    fmt_out = v4l2_format()
    fmt_out.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
    fmt_out.fmt = v4l2_format_union()
    fmt_out_mp = v4l2_pix_format_mplane()
    fmt_out_mp.width = self.out_width
    fmt_out_mp.height = self.out_height
    fmt_out_mp.pixelformat = out_pix
    fmt_out_mp.num_planes = 1
    fmt_out.fmt.pix_mp = fmt_out_mp
    fcntl.ioctl(self._fd, VIDIOC_S_FMT, fmt_out)

    # output (raw) frame rate
    sparm = v4l2_streamparm()
    sparm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    sparm.parm = v4l2_streamparm_parm()
    op = v4l2_outputparm()
    op.timeperframe = v4l2_fract(1, self.encoder_info.fps)
    sparm.parm.output = op
    fcntl.ioctl(self._fd, VIDIOC_S_PARM, sparm)

    # output (raw) format NV12
    fmt_in = v4l2_format()
    fmt_in.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    fmt_in.fmt = v4l2_format_union()
    fmt_in_mp = v4l2_pix_format_mplane()
    fmt_in_mp.width = self.in_width
    fmt_in_mp.height = self.in_height
    fmt_in_mp.pixelformat = V4L2_PIX_FMT_NV12
    fmt_in_mp.num_planes = 1
    fmt_in.fmt.pix_mp = fmt_in_mp
    fcntl.ioctl(self._fd, VIDIOC_S_FMT, fmt_in)

    # request buffers: capture (encoded) via MMAP
    req_cap = v4l2_requestbuffers()
    req_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
    req_cap.memory = V4L2_MEMORY_MMAP
    req_cap.count = 6
    fcntl.ioctl(self._fd, VIDIOC_REQBUFS, req_cap)

    # request buffers: output (raw) via USERPTR
    req_out = v4l2_requestbuffers()
    req_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    req_out.memory = V4L2_MEMORY_USERPTR
    req_out.count = 7
    fcntl.ioctl(self._fd, VIDIOC_REQBUFS, req_out)

    # map capture buffers and queue them
    self._cap_buf_count = req_cap.count
    self._cap_mmaps: list[mmap.mmap] = []
    self._cap_lengths: list[int] = []
    self._cap_planes: list[v4l2_plane] = []
    for i in range(self._cap_buf_count):
      plane = v4l2_plane()
      buf = v4l2_buffer()
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
      buf.index = i
      buf.memory = V4L2_MEMORY_MMAP
      buf.length = 1
      m = v4l2_buffer_m()
      m.planes = C.pointer(plane)
      buf.m = m
      fcntl.ioctl(self._fd, VIDIOC_QUERYBUF, buf)
      length = plane.length
      offset = plane.m.mem_offset
      mm = mmap.mmap(self._fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
      self._cap_mmaps.append(mm)
      self._cap_lengths.append(length)
      self._cap_planes.append(plane)

      # queue capture buffer
      plane_q = v4l2_plane()
      plane_q.length = length
      planes_ptr = C.pointer(plane_q)
      qbuf = v4l2_buffer()
      qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
      qbuf.index = i
      qbuf.memory = V4L2_MEMORY_MMAP
      qbuf.length = 1
      qbuf.m = v4l2_buffer_m()
      qbuf.m.planes = planes_ptr
      fcntl.ioctl(self._fd, VIDIOC_QBUF, qbuf)

    # stream on
    fcntl.ioctl(self._fd, VIDIOC_STREAMON, C.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE))
    fcntl.ioctl(self._fd, VIDIOC_STREAMON, C.c_uint32(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE))

    self.segment_num = getattr(self, "segment_num", -1) + 1
    self.counter = 0
    self.is_open = True
    self._free_in: list[int] = list(range(req_out.count))
    self._free_lock = threading.Lock()
    self._free_cv = threading.Condition(self._free_lock)
    self._extras: list[tuple[int,int,int]] = []  # (frame_id, ts_sof, ts_eof)
    self._extras_lock = threading.Lock()
    self._header: bytes | None = None

    def dq_thread() -> None:
      poller = select.poll()
      POLLIN = 0x001
      POLLOUT = 0x004
      poller.register(self._fd, POLLIN | POLLOUT)
      while self.is_open:
        try:
          events = poller.poll(1000)
        except Exception:
          continue
        if not events:
          # timeout; continue
          continue
        for _fd, revents in events:
          if revents & POLLIN:
            # capture dequeue
            plane = v4l2_plane()
            buf = v4l2_buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            buf.memory = V4L2_MEMORY_MMAP
            buf.length = 1
            buf.m = v4l2_buffer_m()
            buf.m.planes = C.pointer(plane)
            try:
              fcntl.ioctl(self._fd, VIDIOC_DQBUF, buf)
            except Exception:
              continue
            idx = buf.index
            bytesused = plane.bytesused
            flags = buf.flags
            ts = buf.timestamp
            # fetch payload
            payload = self._cap_mmaps[idx][:bytesused]
            # handle flags
            if flags & V4L2_QCOM_BUF_FLAG_EOS:
              self.is_open = False
            elif flags & V4L2_QCOM_BUF_FLAG_CODECCONFIG:
              self._header = bytes(payload)
            else:
              # pop corresponding extras
              with self._extras_lock:
                if self._extras:
                  frame_id, ts_sof, ts_eof = self._extras.pop(0)
                else:
                  frame_id, ts_sof, ts_eof = 0, 0, (ts.tv_sec * 1_000_000 + ts.tv_usec) * 1000
              # publish
              msg = messaging.new_message(self.encoder_info.publish_name)
              ed = getattr(msg, self.encoder_info.publish_name)
              ed.unixTimestampNanos = int(time.time_ns())
              idxs = ed.idx
              idxs.frameId = int(frame_id)
              idxs.timestampSof = int(ts_sof)
              idxs.timestampEof = int(ts_eof)
              idxs.encodeId = self.encode_cnt
              idxs.segmentNum = int(self.segment_num)
              idxs.segmentId = int(self.counter)
              if flags & V4L2_BUF_FLAG_KEYFRAME:
                idxs.flags = 8
                if self._header is not None:
                  ed.header = self._header
              else:
                idxs.flags = 0
              ed.data = bytes(payload)
              idxs.len = len(ed.data)
              ed.width = self.out_width
              ed.height = self.out_height
              self.pm.send(self.encoder_info.publish_name, msg)
              self.counter += 1
              self.encode_cnt += 1
            # requeue capture buffer
            plane_q = v4l2_plane()
            plane_q.length = self._cap_lengths[idx]
            qbuf = v4l2_buffer()
            qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            qbuf.index = idx
            qbuf.memory = V4L2_MEMORY_MMAP
            qbuf.length = 1
            qbuf.m = v4l2_buffer_m()
            qbuf.m.planes = C.pointer(plane_q)
            try:
              fcntl.ioctl(self._fd, VIDIOC_QBUF, qbuf)
            except Exception:
              continue

          if revents & POLLOUT:
            # output dequeue (free one slot)
            plane_o = v4l2_plane()
            buf_o = v4l2_buffer()
            buf_o.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
            buf_o.memory = V4L2_MEMORY_USERPTR
            buf_o.length = 1
            buf_o.m = v4l2_buffer_m()
            buf_o.m.planes = C.pointer(plane_o)
            try:
              fcntl.ioctl(self._fd, VIDIOC_DQBUF, buf_o)
              with self._free_cv:
                self._free_in.append(buf_o.index)
                self._free_cv.notify()
            except Exception:
              pass

    self._dq_thread = threading.Thread(target=dq_thread, name=f"dq-{self.encoder_info.publish_name}", daemon=True)
    self._dq_thread.start()

  def close(self) -> None:
    import fcntl
    import os
    import ctypes as C
    from openpilot.system.loggerd.v4l2_ctypes import (
      VIDIOC_STREAMOFF, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
      VIDIOC_ENCODER_CMD, v4l2_encoder_cmd, V4L2_ENC_CMD_STOP,
    )
    if getattr(self, "is_open", False):
      try:
        # request encoder stop
        cmd = v4l2_encoder_cmd()
        cmd.cmd = V4L2_ENC_CMD_STOP
        fcntl.ioctl(self._fd, VIDIOC_ENCODER_CMD, cmd)
      except Exception:
        pass
      try:
        self._dq_thread.join(timeout=1.0)
      except Exception:
        pass
      try:
        fcntl.ioctl(self._fd, VIDIOC_STREAMOFF, C.c_uint32(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE))
        fcntl.ioctl(self._fd, VIDIOC_STREAMOFF, C.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE))
      except Exception:
        pass
      try:
        os.close(self._fd)
      except Exception:
        pass
      self.is_open = False

  def encode_frame(self, buf: VisionBuf, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> int:
    import fcntl
    import ctypes as C
    import numpy as np
    from openpilot.system.loggerd.v4l2_ctypes import (
      VIDIOC_QBUF, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, V4L2_MEMORY_USERPTR,
      V4L2_BUF_FLAG_TIMESTAMP_COPY, v4l2_buffer, v4l2_plane, v4l2_buffer_m, timeval,
    )

    # wait for a free input slot
    with self._free_cv:
      while not self._free_in:
        self._free_cv.wait(timeout=0.01)
        if not self.is_open:
          return -1
      index = self._free_in.pop(0)

    # build plane for USERPTR
    data = np.asarray(buf.data, dtype=np.uint8)
    length = data.size
    plane = v4l2_plane()
    plane.length = length
    plane.bytesused = length
    plane.m.userptr = C.c_ulong(int(data.ctypes.data))
    plane.reserved[0] = buf.fd

    tv = timeval()
    tv.tv_sec = int(timestamp_eof // 1_000_000_000)
    tv.tv_usec = int((timestamp_eof // 1000) % 1_000_000)

    vbuf = v4l2_buffer()
    vbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    vbuf.index = index
    vbuf.memory = V4L2_MEMORY_USERPTR
    vbuf.length = 1
    vbuf.flags = V4L2_BUF_FLAG_TIMESTAMP_COPY
    vbuf.timestamp = tv
    vbuf.m = v4l2_buffer_m()
    vbuf.m.planes = C.pointer(plane)

    # track extras for publishing when capture buffer arrives
    with self._extras_lock:
      self._extras.append((frame_id, timestamp_sof, timestamp_eof))

    try:
      fcntl.ioctl(self._fd, VIDIOC_QBUF, vbuf)
    except Exception:
      return -1
    return self.counter


class JpegThumbnail:
  def __init__(self, publish_name: str, width: int, height: int):
    self.publish_name = publish_name
    self.width = width
    self.height = height
    self.pm = messaging.PubMaster([publish_name])

  def push(self, buf: VisionBuf, frame_id: int, timestamp_eof: int) -> None:
    try:
      import numpy as np
      import jpeglib
    except Exception:
      # If jpeglib is unavailable, skip publishing to avoid blocking
      return

    W, H, S = buf.width, buf.height, buf.stride
    # Extract Y plane
    y_full = np.asarray(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, S))[:H, :W]
    # Extract interleaved UV plane
    uv_rows = H // 2
    uv_full = np.asarray(buf.data[buf.uv_offset:buf.uv_offset + uv_rows * S], dtype=np.uint8).reshape((uv_rows, S))[:, :W]

    # Downscale by 2 in each dimension (quarter resolution area)
    tw, th = self.width, self.height
    assert W % 2 == 0 and H % 2 == 0 and tw * 2 == W and th * 2 == H

    y_small = y_full.reshape(th, 2, tw, 2).mean(axis=(1, 3)).astype(np.uint8)
    # For UV, pick every other sample to approximate 4:2:0
    # uv_full has shape (H/2, W), where each row contains interleaved U,V bytes per column
    u_full = uv_full[:, 0::2]
    v_full = uv_full[:, 1::2]
    # Downscale U and V to (th/2, tw/2)
    u_small = u_full.reshape(th // 2, 2, tw // 2, 2).mean(axis=(1, 3)).astype(np.uint8)
    v_small = v_full.reshape(th // 2, 2, tw // 2, 2).mean(axis=(1, 3)).astype(np.uint8)

    # Encode as JPEG from YUV420
    jpeg_bytes = jpeglib.encode_YUV420(y_small, u_small, v_small, sampling="420", quality=80)

    msg = messaging.new_message("thumbnail")
    msg.thumbnail.frameId = int(frame_id)
    msg.thumbnail.timestampEof = int(timestamp_eof)
    msg.thumbnail.thumbnail = jpeg_bytes
    # msg.thumbnail.encoding left default (jpeg)
    self.pm.send(self.publish_name, msg)


# ------------------------------ Worker thread --------------------------------

def encoder_thread(state: EncoderdState, cam_info: LogCameraInfo) -> None:
  vipc = VisionIpcClient("camerad", cam_info.stream_type, False)

  encoders: list[BaseEncoder] = []
  jpeg: JpegThumbnail | None = None
  cur_seg = 0
  frames_per_seg = SEGMENT_LENGTH * MAIN_FPS

  while True:
    if not vipc.connect(False):
      time.sleep(0.005)
      continue

    # init encoders on first connection
    if not encoders:
      assert vipc.num_buffers
      in_w, in_h = vipc.width, vipc.height
      assert in_w and in_h and in_w > 0 and in_h > 0

      for ei in cam_info.encoder_infos:
        EncoderCls = V4LEncoder if not PC() else FfmpegEncoder
        e = EncoderCls(ei, in_w, in_h)
        e.open()
        encoders.append(e)

      if cam_info.encoder_infos and cam_info.encoder_infos[0].thumbnail_name:
        jpeg = JpegEncoder(cam_info.encoder_infos[0].thumbnail_name, in_w // 4, in_h // 4)

    # main recv/encode loop
    while True:
      buf = vipc.recv(100)
      if buf is None:
        continue

      frame_id = vipc.frame_id
      timestamp_sof = vipc.timestamp_sof
      timestamp_eof = vipc.timestamp_eof

      if not sync_encoders(state, cam_info.stream_type, frame_id):
        continue

      # rotate when segment boundary is crossed
      if cur_seg >= 0 and frame_id >= ((cur_seg + 1) * frames_per_seg) + state.start_frame_id:
        for e in encoders:
          e.close()
          e.open()
        cur_seg += 1

      # encode
      for e in encoders:
        out_id = e.encode_frame(buf, frame_id, timestamp_sof, timestamp_eof)
        if out_id == -1:
          # minimal logging; keep parity with C++
          pass

      if jpeg and (frame_id % 1200 == 100):
        extra = SimpleNamespace(frame_id=frame_id, timestamp_eof=timestamp_eof)
        jpeg.pushThumbnail(buf, extra)


# ----------------------------- Camera definitions -----------------------------

def build_camera_infos(stream: bool) -> list[LogCameraInfo]:
  # Mirror loggerd.h encoder infos
  def main_info(name_pub: str, filename: str | None, thumbnail: str | None = None) -> EncoderInfo:
    return EncoderInfo(
      publish_name=name_pub,
      filename=filename,
      thumbnail_name=thumbnail,
      get_settings=lambda w: EncoderSettings.main_encoder_settings(w),
    )

  def qcam_info() -> EncoderInfo:
    return EncoderInfo(
      publish_name="qRoadEncodeData",
      filename="qcamera.ts",
      include_audio=False,  # follow params if needed
      frame_width=526,
      frame_height=330,
      get_settings=lambda _w: EncoderSettings.qcam_encoder_settings(),
    )

  def stream_info(name_pub: str) -> EncoderInfo:
    return EncoderInfo(
      publish_name=name_pub,
      filename=None,
      record=False,
      get_settings=lambda _w: EncoderSettings.stream_encoder_settings(),
    )

  if stream:
    return [
      LogCameraInfo("road_cam_encoder", VisionStreamType.VISION_STREAM_ROAD, encoder_infos=[stream_info("livestreamRoadEncodeData")]),
      LogCameraInfo("wide_road_cam_encoder", VisionStreamType.VISION_STREAM_WIDE_ROAD, encoder_infos=[stream_info("livestreamWideRoadEncodeData")]),
      LogCameraInfo("driver_cam_encoder", VisionStreamType.VISION_STREAM_DRIVER, encoder_infos=[stream_info("livestreamDriverEncodeData")]),
    ]
  else:
    road = LogCameraInfo(
      "road_cam_encoder",
      VisionStreamType.VISION_STREAM_ROAD,
      encoder_infos=[
        main_info("roadEncodeData", "fcamera.hevc", "thumbnail"),
        qcam_info(),
      ],
    )
    wide = LogCameraInfo("wide_road_cam_encoder", VisionStreamType.VISION_STREAM_WIDE_ROAD, encoder_infos=[main_info("wideRoadEncodeData", "ecamera.hevc")])
    drv = LogCameraInfo("driver_cam_encoder", VisionStreamType.VISION_STREAM_DRIVER, encoder_infos=[main_info("driverEncodeData", "dcamera.hevc")])
    return [road, wide, drv]


# ----------------------------------- Main ------------------------------------

shutdown_event = threading.Event()


def encoderd_thread(cameras: list[LogCameraInfo]) -> None:
  state = EncoderdState()

  # Wait for any stream to appear
  streams = set()
  while not shutdown_event.is_set():
    streams = VisionIpcClient.available_streams("camerad", block=False)
    if streams:
      break
    time.sleep(0.1)

  if not streams:
    return

  # Start threads for available streams only
  threads: list[threading.Thread] = []
  for st in streams:
    # map VisionStreamType to LogCameraInfo
    match = next((ci for ci in cameras if ci.stream_type == st), None)
    if match is None:
      continue
    state.max_waiting += 1
    t = threading.Thread(target=encoder_thread, args=(state, match), name=match.thread_name, daemon=True)
    t.start()
    threads.append(t)

  for t in threads:
    while t.is_alive():
      t.join(timeout=0.2)
      if shutdown_event.is_set():
        break


def main() -> int:
  parser = argparse.ArgumentParser(description="encoderd (python)")
  parser.add_argument("--stream", action="store_true", help="use livestream encoders")
  args = parser.parse_args()

  if not PC():
    # Best-effort realtime/affinity; ignore failures
    try:
      os.sched_setaffinity(0, {3})
    except Exception:
      pass

  def _sigterm(_signum, _frame):
    shutdown_event.set()

  signal.signal(signal.SIGINT, _sigterm)
  signal.signal(signal.SIGTERM, _sigterm)

  cameras = build_camera_infos(stream=args.stream)
  encoderd_thread(cameras)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
