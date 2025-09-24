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

Backends are implemented in separate modules and selected by platform.
"""
import argparse
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable

from msgq.visionipc import VisionIpcClient, VisionStreamType
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
      etype = "BIG_BOX_LOSSLESS" if PC else "FULL_H_E_V_C"
      return EncoderSettings(etype, 5_000_000, 20)
    else:
      etype = "BIG_BOX_LOSSLESS" if PC else "FULL_H_E_V_C"
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


# ------------------------------ Worker thread --------------------------------

def encoder_thread(state: EncoderdState, cam_info: LogCameraInfo) -> None:
  vipc = VisionIpcClient("camerad", cam_info.stream_type, False)

  encoders: list = []
  jpeg: JpegEncoder | None = None
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
        EncoderCls = V4LEncoder if not PC else FfmpegEncoder
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

  if not PC:
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
