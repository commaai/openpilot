#!/usr/bin/env python3
import os
import signal
import sys
import threading
import time
import traceback
from functools import cache

from msgq.visionipc import VisionIpcClient, VisionStreamType

from openpilot.common.hardware import PC, TICI
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.encoder.encoder import MAIN_FPS, EncoderInfo, EncoderSettings, FrameExtra, LogCameraInfo
from openpilot.system.loggerd.encoder.jpeg_encoder import JpegEncoder

if TICI:
  from openpilot.system.loggerd.encoder.v4l_encoder import V4LEncoder as Encoder
else:
  from openpilot.system.loggerd.encoder.ffmpeg_encoder import FfmpegEncoder as Encoder

main_road_encoder_info = EncoderInfo(
  publish_name="roadEncodeData",
  thumbnail_name="thumbnail",
  get_settings=EncoderSettings.main_encoder_settings,
)

main_wide_road_encoder_info = EncoderInfo(
  publish_name="wideRoadEncodeData",
  get_settings=EncoderSettings.main_encoder_settings,
)

main_driver_encoder_info = EncoderInfo(
  publish_name="driverEncodeData",
  get_settings=EncoderSettings.main_encoder_settings,
)

qcam_encoder_info = EncoderInfo(
  publish_name="qRoadEncodeData",
  frame_width=526,
  frame_height=330,
  get_settings=EncoderSettings.qcam_encoder_settings,
)

stream_road_encoder_info = EncoderInfo(
  publish_name="livestreamRoadEncodeData",
  is_live=True,
  get_settings=EncoderSettings.stream_encoder_settings,
)

stream_wide_road_encoder_info = EncoderInfo(
  publish_name="livestreamWideRoadEncodeData",
  is_live=True,
  get_settings=EncoderSettings.stream_encoder_settings,
)

stream_driver_encoder_info = EncoderInfo(
  publish_name="livestreamDriverEncodeData",
  is_live=True,
  get_settings=EncoderSettings.stream_encoder_settings,
)

cameras_logged = [
  LogCameraInfo(thread_name="road_cam_encoder", stream_type=VisionStreamType.VISION_STREAM_ROAD,
                encoder_infos=[main_road_encoder_info, qcam_encoder_info]),
  LogCameraInfo(thread_name="wide_road_cam_encoder", stream_type=VisionStreamType.VISION_STREAM_WIDE_ROAD,
                encoder_infos=[main_wide_road_encoder_info]),
  LogCameraInfo(thread_name="driver_cam_encoder", stream_type=VisionStreamType.VISION_STREAM_DRIVER,
                encoder_infos=[main_driver_encoder_info]),
]

stream_cameras_logged = [
  LogCameraInfo(thread_name="road_cam_encoder", stream_type=VisionStreamType.VISION_STREAM_ROAD,
                encoder_infos=[stream_road_encoder_info]),
  LogCameraInfo(thread_name="wide_road_cam_encoder", stream_type=VisionStreamType.VISION_STREAM_WIDE_ROAD,
                encoder_infos=[stream_wide_road_encoder_info]),
  LogCameraInfo(thread_name="driver_cam_encoder", stream_type=VisionStreamType.VISION_STREAM_DRIVER,
                encoder_infos=[stream_driver_encoder_info]),
]


class EncoderdState:
  def __init__(self):
    self.lock = threading.Lock()
    self.max_waiting = 0

    # sync logic for startup
    self.encoders_ready = 0
    self.start_frame_id = 0
    self.camera_ready: dict[int, bool] = {}
    self.camera_synced: dict[int, bool] = {}


def sync_encoders(s: EncoderdState, cam_type: int, frame_id: int) -> bool:
  """Handle initial encoder syncing by waiting for all encoders to reach the same frame id."""
  with s.lock:
    if s.camera_synced.get(cam_type, False):
      return True

    if s.max_waiting > 1 and s.encoders_ready != s.max_waiting:
      # add a small margin to the start frame id in case one of the encoders already dropped the next frame
      s.start_frame_id = max(s.start_frame_id, frame_id + 2)
      if not s.camera_ready.get(cam_type, False):
        s.camera_ready[cam_type] = True
        s.encoders_ready += 1
        cloudlog.debug(f"camera {cam_type} encoder ready")
      return False
    else:
      if s.max_waiting == 1:
        s.start_frame_id = max(s.start_frame_id, frame_id)
      synced = frame_id >= s.start_frame_id
      s.camera_synced[cam_type] = synced
      if not synced:
        cloudlog.debug(f"camera {cam_type} waiting for frame {s.start_frame_id}, cur {frame_id}")
      return synced


@cache
def _params() -> Params:
  return Params()


def encoder_set_bitrate(e) -> None:
  val = _params().get("LivestreamEncoderBitrate")
  if val is None:
    return
  e.set_bitrate(int(val))


def encoder_request_keyframe(e) -> None:
  if _params().get_bool("LivestreamRequestKeyframe"):
    e.request_keyframe()


def encoder_thread(s: EncoderdState, cam_info: LogCameraInfo, do_exit: threading.Event) -> None:
  segment_length = int(os.getenv("LOGGERD_SEGMENT_LENGTH", "60")) if os.getenv("LOGGERD_TEST") else 60

  encoders = []
  jpeg_encoder = None
  vipc_client = VisionIpcClient("camerad", cam_info.stream_type, False)

  cur_seg = 0
  try:
    while not do_exit.is_set():
      if not vipc_client.connect(False):
        time.sleep(0.005)
        continue

      # init encoders
      if not encoders:
        width, height = vipc_client.width, vipc_client.height
        cloudlog.warning(f"encoder {cam_info.thread_name} init {width}x{height}")
        assert width > 0 and height > 0

        for encoder_info in cam_info.encoder_infos:
          e = Encoder(encoder_info, width, height)
          e.encoder_open()
          encoders.append(e)

        # only one thumbnail can be generated per camera stream
        if cam_info.encoder_infos[0].thumbnail_name is not None:
          jpeg_encoder = JpegEncoder(cam_info.encoder_infos[0].thumbnail_name, width // 4, height // 4)

      lagging = False
      while not do_exit.is_set():
        buf = vipc_client.recv()
        if buf is None:
          continue
        extra = FrameExtra(vipc_client.frame_id, vipc_client.timestamp_sof, vipc_client.timestamp_eof)

        # detect loop around and drop the frames
        if buf.frame_id != extra.frame_id:
          if not lagging:
            cloudlog.error(f"encoder {cam_info.thread_name} lag  buffer id: {buf.frame_id} extra id: {extra.frame_id}")
            lagging = True
          continue
        lagging = False

        if not sync_encoders(s, cam_info.stream_type, extra.frame_id):
          continue
        if do_exit.is_set():
          break

        # do rotation if required
        frames_per_seg = segment_length * MAIN_FPS
        if cur_seg >= 0 and extra.frame_id >= ((cur_seg + 1) * frames_per_seg) + s.start_frame_id:
          for e in encoders:
            e.encoder_close()
            e.encoder_open()
          cur_seg += 1

        # encode a frame
        for i, e in enumerate(encoders):
          if cam_info.encoder_infos[i].is_live:
            encoder_set_bitrate(e)
            encoder_request_keyframe(e)

          out_id = e.encode_frame(buf, extra)

          if out_id == -1:
            cloudlog.error(f"Failed to encode frame. frame_id: {extra.frame_id}")

        if jpeg_encoder is not None and extra.frame_id % 1200 == 100:
          jpeg_encoder.push_thumbnail(buf, extra)
  finally:
    for e in encoders:
      try:
        e.close()
      except OSError:
        cloudlog.exception(f"encoder {cam_info.thread_name} close failed")


def encoderd_thread(cameras: list[LogCameraInfo], do_exit: threading.Event) -> None:
  s = EncoderdState()

  streams: set[int] = set()
  while not do_exit.is_set():
    streams = VisionIpcClient.available_streams("camerad", False)
    if streams:
      break
    time.sleep(0.1)

  if streams:
    encoder_threads = []
    for stream in streams:
      cam_info = next(cam for cam in cameras if cam.stream_type == stream)
      s.max_waiting += 1
      encoder_threads.append(threading.Thread(target=encoder_thread, args=(s, cam_info, do_exit), name=cam_info.thread_name))

    for t in encoder_threads:
      t.start()
    for t in encoder_threads:
      while t.is_alive():
        t.join(0.1)


def _thread_excepthook(args) -> None:
  # an encoder thread dying leaves streams silently missing; die loudly like the C++ asserts did and let manager restart us
  traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
  cloudlog.error(f"encoderd thread {args.thread.name if args.thread else '?'} crashed: {args.exc_value}")
  os._exit(1)


def main(stream: bool = False) -> None:
  if not PC:
    config_realtime_process([3], 52)
  threading.excepthook = _thread_excepthook

  do_exit = threading.Event()
  signal.signal(signal.SIGINT, lambda sig, frame: do_exit.set())
  signal.signal(signal.SIGTERM, lambda sig, frame: do_exit.set())

  cameras = stream_cameras_logged if (stream or "--stream" in sys.argv[1:]) else cameras_logged
  encoderd_thread(cameras, do_exit)


if __name__ == "__main__":
  main()
