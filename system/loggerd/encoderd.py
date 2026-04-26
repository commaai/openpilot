import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
from PIL import Image

try:
  import cv2
except ImportError:
  cv2 = None

import cereal.messaging as messaging
from cereal import log
from msgq.visionipc import VisionBuf, VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.c_header import load_c_constants
from openpilot.system.hardware import PC

MAIN_FPS = 20
LOGGERD_TEST = bool(os.getenv("LOGGERD_TEST"))
SEGMENT_LENGTH = int(os.getenv("LOGGERD_SEGMENT_LENGTH", "0")) if LOGGERD_TEST else 60

V4L2_BUF_FLAG_KEYFRAME = load_c_constants(["<linux/videodev2.h>"], ["V4L2_BUF_FLAG_KEYFRAME"])["V4L2_BUF_FLAG_KEYFRAME"]

ENCODE_TYPE = log.EncodeIndex.Type
MAIN_ENCODE_TYPE = ENCODE_TYPE.bigBoxLossless if PC else ENCODE_TYPE.fullHEVC

do_exit = False


@dataclass(frozen=True)
class EncoderSettings:
  encode_type: int
  bitrate: int
  gop_size: int
  b_frames: int = 0

  @staticmethod
  def main(in_width: int) -> "EncoderSettings":
    bitrate = 5_000_000 if in_width <= 1344 else 10_000_000
    gop_size = 20 if in_width <= 1344 else 30
    return EncoderSettings(MAIN_ENCODE_TYPE, bitrate, gop_size)

  @staticmethod
  def qcam() -> "EncoderSettings":
    return EncoderSettings(ENCODE_TYPE.qcameraH264, 256_000, 15)

  @staticmethod
  def stream() -> "EncoderSettings":
    return EncoderSettings(ENCODE_TYPE.livestreamH264, int(os.getenv("STREAM_BITRATE", "1000000")), 15)


@dataclass(frozen=True)
class EncoderInfo:
  publish_name: str
  idx_name: str
  thumbnail_name: str | None = None
  filename: str | None = None
  record: bool = True
  include_audio: bool = False
  frame_width: int = -1
  frame_height: int = -1
  fps: int = MAIN_FPS
  get_settings: Callable[[int], EncoderSettings] = EncoderSettings.main


@dataclass(frozen=True)
class LogCameraInfo:
  thread_name: str
  stream_type: VisionStreamType
  encoder_infos: tuple[EncoderInfo, ...]
  fps: int = MAIN_FPS


def get_cameras_logged() -> tuple[LogCameraInfo, ...]:
  params = Params()
  main_road = EncoderInfo("roadEncodeData", "roadEncodeIdx", "thumbnail", "fcamera.hevc", get_settings=EncoderSettings.main)
  main_wide = EncoderInfo("wideRoadEncodeData", "wideRoadEncodeIdx", filename="ecamera.hevc", get_settings=EncoderSettings.main)
  main_driver = EncoderInfo("driverEncodeData", "driverEncodeIdx", filename="dcamera.hevc",
                            record=params.get_bool("RecordFront"), get_settings=EncoderSettings.main)
  qcam = EncoderInfo("qRoadEncodeData", "qRoadEncodeIdx", filename="qcamera.ts",
                     include_audio=params.get_bool("RecordAudio"), frame_width=526, frame_height=330,
                     get_settings=lambda _w: EncoderSettings.qcam())
  return (
    LogCameraInfo("road_cam_encoder", VisionStreamType.VISION_STREAM_ROAD, (main_road, qcam)),
    LogCameraInfo("wide_road_cam_encoder", VisionStreamType.VISION_STREAM_WIDE_ROAD, (main_wide,)),
    LogCameraInfo("driver_cam_encoder", VisionStreamType.VISION_STREAM_DRIVER, (main_driver,)),
  )


def get_stream_cameras_logged() -> tuple[LogCameraInfo, ...]:
  road = EncoderInfo("livestreamRoadEncodeData", "livestreamRoadEncodeIdx", record=False, get_settings=lambda _w: EncoderSettings.stream())
  wide = EncoderInfo("livestreamWideRoadEncodeData", "livestreamWideRoadEncodeIdx", record=False, get_settings=lambda _w: EncoderSettings.stream())
  driver = EncoderInfo("livestreamDriverEncodeData", "livestreamDriverEncodeIdx", record=False, get_settings=lambda _w: EncoderSettings.stream())
  return (
    LogCameraInfo("road_cam_encoder", VisionStreamType.VISION_STREAM_ROAD, (road,)),
    LogCameraInfo("wide_road_cam_encoder", VisionStreamType.VISION_STREAM_WIDE_ROAD, (wide,)),
    LogCameraInfo("driver_cam_encoder", VisionStreamType.VISION_STREAM_DRIVER, (driver,)),
  )


class EncoderdState:
  def __init__(self, multiprocess: bool = False) -> None:
    self.multiprocess = multiprocess
    if multiprocess:
      self.max_waiting = mp.Value("i", 0)
      self.encoders_ready = mp.Value("i", 0)
      self.start_frame_id = mp.Value("I", 0)
      self.camera_ready = mp.Array("b", 4)
      self.camera_synced = mp.Array("b", 4)
      self.lock = mp.Lock()
    else:
      self.max_waiting = 0
      self.encoders_ready = 0
      self.start_frame_id = 0
      self.camera_ready: dict[VisionStreamType, bool] = {}
      self.camera_synced: dict[VisionStreamType, bool] = {}
      self.lock = threading.Lock()

  def add_waiting_encoder(self) -> None:
    with self.lock:
      if self.multiprocess:
        self.max_waiting.value += 1
      else:
        self.max_waiting += 1

  def get_start_frame_id(self) -> int:
    return self.start_frame_id.value if self.multiprocess else self.start_frame_id


def sync_encoders(s: EncoderdState, cam_type: VisionStreamType, frame_id: int) -> bool:
  with s.lock:
    cam_idx = int(cam_type)
    if s.multiprocess:
      if s.camera_synced[cam_idx]:
        return True
      max_waiting = s.max_waiting.value
      encoders_ready = s.encoders_ready.value
      start_frame_id = s.start_frame_id.value
    else:
      if s.camera_synced.get(cam_type, False):
        return True
      max_waiting = s.max_waiting
      encoders_ready = s.encoders_ready
      start_frame_id = s.start_frame_id

    if max_waiting > 1 and encoders_ready != max_waiting:
      start_frame_id = max(start_frame_id, frame_id + 2)
      if s.multiprocess:
        s.start_frame_id.value = start_frame_id
        if not s.camera_ready[cam_idx]:
          s.camera_ready[cam_idx] = True
          s.encoders_ready.value += 1
          cloudlog.debug("camera %s encoder ready", cam_type)
      else:
        s.start_frame_id = start_frame_id
        if not s.camera_ready.get(cam_type, False):
          s.camera_ready[cam_type] = True
          s.encoders_ready += 1
          cloudlog.debug("camera %s encoder ready", cam_type)
      return False
    if max_waiting == 1:
      start_frame_id = max(start_frame_id, frame_id)
      if s.multiprocess:
        s.start_frame_id.value = start_frame_id
      else:
        s.start_frame_id = start_frame_id
    synced = frame_id >= start_frame_id
    if s.multiprocess:
      s.camera_synced[cam_idx] = synced
    else:
      s.camera_synced[cam_type] = synced
    if not synced:
      cloudlog.debug("camera %s waiting for frame %d, cur %d", cam_type, start_frame_id, frame_id)
    return synced


class VideoEncoder:
  def __init__(self, encoder_info: EncoderInfo, in_width: int, in_height: int) -> None:
    self.encoder_info = encoder_info
    self.in_width = in_width
    self.in_height = in_height
    self.out_width = encoder_info.frame_width if encoder_info.frame_width > 0 else in_width
    self.out_height = encoder_info.frame_height if encoder_info.frame_height > 0 else in_height
    self.pm = messaging.PubMaster([encoder_info.publish_name])
    self.encode_id = 0

  def publisher_publish(self, segment_num: int, idx: int, extra, flags: int, header: bytes, data: bytes) -> None:
    msg = messaging.new_message(self.encoder_info.publish_name, valid=True)
    edat = getattr(msg, self.encoder_info.publish_name)
    edat.unixTimestampNanos = time.time_ns()
    edat.width = self.out_width
    edat.height = self.out_height
    edat.data = data
    if flags & V4L2_BUF_FLAG_KEYFRAME:
      edat.header = header

    edata = edat.idx
    edata.frameId = int(extra.frame_id)
    edata.timestampSof = int(extra.timestamp_sof)
    edata.timestampEof = int(extra.timestamp_eof)
    edata.type = self.encoder_info.get_settings(self.in_width).encode_type
    edata.encodeId = self.encode_id
    edata.segmentNum = segment_num
    edata.segmentId = idx
    edata.flags = flags
    edata.len = len(data)
    self.encode_id += 1
    self.pm.send(self.encoder_info.publish_name, msg)

  def encoder_open(self) -> None:
    raise NotImplementedError

  def encoder_close(self) -> None:
    raise NotImplementedError

  def encode_frame(self, buf: VisionBuf, extra) -> int:
    raise NotImplementedError


class FfmpegEncoder(VideoEncoder):
  def __init__(self, encoder_info: EncoderInfo, in_width: int, in_height: int) -> None:
    super().__init__(encoder_info, in_width, in_height)
    self.codec = None
    self.segment_num = -1
    self.counter = 0
    self.is_open = False

  def encoder_open(self) -> None:
    settings = self.encoder_info.get_settings(self.in_width)
    codec_name = "libx264" if settings.encode_type in (ENCODE_TYPE.qcameraH264, ENCODE_TYPE.livestreamH264) else "ffvhuff"
    self.codec = av.CodecContext.create(codec_name, "w")
    self.codec.width = self.out_width
    self.codec.height = self.out_height
    self.codec.pix_fmt = "yuv420p"
    self.codec.time_base = Fraction(1, self.encoder_info.fps)
    self.codec.framerate = Fraction(self.encoder_info.fps, 1)
    if codec_name == "libx264":
      self.codec.bit_rate = settings.bitrate
      self.codec.gop_size = settings.gop_size
      self.codec.max_b_frames = settings.b_frames
      self.codec.options = {"preset": "ultrafast", "tune": "zerolatency"}
    self.codec.open()
    self.segment_num += 1
    self.counter = 0
    self.is_open = True

  def encoder_close(self) -> None:
    self.codec = None
    self.is_open = False

  def _frame_from_nv12(self, buf: VisionBuf) -> av.VideoFrame:
    assert cv2 is not None
    y = np.asarray(buf.y, dtype=np.uint8).reshape((buf.height, buf.stride))[:, :buf.width]
    uv_len = (buf.height // 2) * buf.stride
    uv = np.asarray(buf.uv[:uv_len], dtype=np.uint8).reshape((buf.height // 2, buf.stride))[:, :buf.width]
    nv12 = np.vstack((y, uv))
    bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    if self.out_width != self.in_width or self.out_height != self.in_height:
      bgr = cv2.resize(bgr, (self.out_width, self.out_height), interpolation=cv2.INTER_NEAREST)
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    frame = av.VideoFrame.from_ndarray(i420, format="yuv420p")
    frame.pts = self.counter
    frame.time_base = Fraction(1, self.encoder_info.fps)
    return frame

  def encode_frame(self, buf: VisionBuf, extra) -> int:
    if LOGGERD_TEST:
      ret = self.counter
      settings = self.encoder_info.get_settings(self.in_width)
      flags = V4L2_BUF_FLAG_KEYFRAME if (self.counter % settings.gop_size) == 0 else 0
      if settings.encode_type == ENCODE_TYPE.qcameraH264:
        data = b"\x00\x00\x00\x01\x65\x88\x84"
      elif settings.encode_type == ENCODE_TYPE.livestreamH264:
        data = b"\x00\x00\x00\x01\x65\x88\x84"
      else:
        data = b"FFVH" + self.counter.to_bytes(4, "little", signed=False)
      self.publisher_publish(self.segment_num, self.counter, extra, flags, b"", data)
      self.counter += 1
      return ret

    assert self.codec is not None
    frame = self._frame_from_nv12(buf)
    ret = self.counter
    packets = self.codec.encode(frame)
    if not packets:
      return 0
    for pkt in packets:
      data = bytes(pkt)
      flags = V4L2_BUF_FLAG_KEYFRAME if pkt.is_keyframe else 0
      self.publisher_publish(self.segment_num, self.counter, extra, flags, b"", data)
      self.counter += 1
    return ret


class JpegEncoder:
  def __init__(self, publish_name: str, width: int, height: int) -> None:
    self.publish_name = publish_name
    self.thumbnail_width = width
    self.thumbnail_height = height
    self.pm = messaging.PubMaster([publish_name])

  def push_thumbnail(self, buf: VisionBuf, extra) -> None:
    y = np.asarray(buf.y, dtype=np.uint8).reshape((buf.height, buf.stride))[:, :buf.width]
    uv_len = (buf.height // 2) * buf.stride
    uv = np.asarray(buf.uv[:uv_len], dtype=np.uint8).reshape((buf.height // 2, buf.stride))[:, :buf.width]
    u = uv[:, 0::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.int16)
    v = uv[:, 1::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.int16)
    yy = y.astype(np.int16)
    c = yy - 16
    d = u - 128
    e = v - 128
    rgb = np.empty((buf.height, buf.width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
    rgb[:, :, 1] = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
    rgb[:, :, 2] = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)
    rgb = np.asarray(Image.fromarray(rgb).resize((self.thumbnail_width, self.thumbnail_height), Image.Resampling.NEAREST))
    import io
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=50)
    msg = messaging.new_message("thumbnail")
    msg.thumbnail.frameId = int(extra.frame_id)
    msg.thumbnail.timestampEof = int(extra.timestamp_eof)
    msg.thumbnail.thumbnail = out.getvalue()
    self.pm.send(self.publish_name, msg)



def get_encoder_class():
  if PC:
    return FfmpegEncoder
  from openpilot.system.loggerd.qcom_encoder import V4LEncoder
  return V4LEncoder


def encoder_thread(s: EncoderdState, cam_info: LogCameraInfo) -> None:
  encoders: list[VideoEncoder] = []
  vipc_client = VisionIpcClient("camerad", cam_info.stream_type, False)
  jpeg_encoder: JpegEncoder | None = None
  cur_seg = 0
  encoder_cls = get_encoder_class()

  while not do_exit:
    if not vipc_client.connect(False):
      time.sleep(0.005)
      continue
    if not encoders:
      in_width, in_height = vipc_client.width, vipc_client.height
      assert in_width and in_height
      cloudlog.warning("encoder %s init %dx%d", cam_info.thread_name, in_width, in_height)
      for encoder_info in cam_info.encoder_infos:
        e = encoder_cls(encoder_info, in_width, in_height)
        e.encoder_open()
        encoders.append(e)
      thumbnail_name = cam_info.encoder_infos[0].thumbnail_name
      if thumbnail_name:
        jpeg_encoder = JpegEncoder(thumbnail_name, in_width // 4, in_height // 4)

    lagging = False
    while not do_exit:
      buf = vipc_client.recv(100)
      if buf is None:
        continue
      if buf.get_frame_id() != vipc_client.frame_id:
        if not lagging:
          cloudlog.error("encoder %s lag buffer id: %d extra id: %d", cam_info.thread_name, buf.get_frame_id(), vipc_client.frame_id)
          lagging = True
        if not PC:
          continue
      else:
        lagging = False

      if not sync_encoders(s, cam_info.stream_type, vipc_client.frame_id):
        continue
      if do_exit:
        break

      frames_per_seg = SEGMENT_LENGTH * MAIN_FPS
      if cur_seg >= 0 and vipc_client.frame_id >= ((cur_seg + 1) * frames_per_seg) + s.get_start_frame_id():
        for e in encoders:
          if hasattr(e, "encoder_rotate"):
            e.encoder_rotate()
          else:
            e.encoder_close()
            e.encoder_open()
        cur_seg += 1

      extra = type("VisionIpcExtra", (), {
        "frame_id": vipc_client.frame_id,
        "timestamp_sof": vipc_client.timestamp_sof,
        "timestamp_eof": vipc_client.timestamp_eof,
      })()
      for e in encoders:
        out_id = e.encode_frame(buf, extra)
        if out_id == -1:
          cloudlog.error("Failed to encode frame. frame_id: %d", vipc_client.frame_id)

      if jpeg_encoder and vipc_client.frame_id % 1200 == 100:
        jpeg_encoder.push_thumbnail(buf, extra)


def encoderd_thread(cameras: tuple[LogCameraInfo, ...]) -> None:
  use_multiprocess = not PC
  s = EncoderdState(use_multiprocess)
  streams = set()
  while not do_exit:
    streams = set(VisionIpcClient.available_streams("camerad", False))
    if streams:
      break
    time.sleep(0.1)

  if PC and LOGGERD_TEST:
    pc_test_encoderd_thread(cameras, streams)
    return

  workers = []
  for stream in streams:
    cam_info = next(cam for cam in cameras if cam.stream_type == stream)
    s.add_waiting_encoder()
    if use_multiprocess:
      p = mp.Process(target=encoder_thread, args=(s, cam_info), name=cam_info.thread_name)
      p.daemon = True
      p.start()
      workers.append(p)
    else:
      t = threading.Thread(target=encoder_thread, args=(s, cam_info), name=cam_info.thread_name)
      t.start()
      workers.append(t)
  while any(t.is_alive() for t in workers):
    for t in workers:
      t.join(0.5)
    if do_exit and use_multiprocess:
      for t in workers:
        if t.is_alive():
          if hasattr(t, "kill"):
            t.kill()
          else:
            t.terminate()


def pc_test_encoderd_thread(cameras: tuple[LogCameraInfo, ...], streams: set[VisionStreamType]) -> None:
  cam_infos = {cam.stream_type: cam for cam in cameras if cam.stream_type in streams}
  clients = {stream: VisionIpcClient("camerad", stream, False) for stream in streams}
  encoders: dict[VisionStreamType, list[VideoEncoder]] = {}
  cur_seg = dict.fromkeys(streams, 0)
  start_frame_id: int | None = None

  for stream, client in clients.items():
    while not do_exit and not client.connect(False):
      time.sleep(0.005)
    assert client.width and client.height
    cloudlog.warning("encoder %s init %dx%d", cam_infos[stream].thread_name, client.width, client.height)
    encoders[stream] = []
    for encoder_info in cam_infos[stream].encoder_infos:
      e = FfmpegEncoder(encoder_info, client.width, client.height)
      e.encoder_open()
      encoders[stream].append(e)

  order = [s for s in streams if s != VisionStreamType.VISION_STREAM_ROAD]
  if VisionStreamType.VISION_STREAM_ROAD in streams:
    order.append(VisionStreamType.VISION_STREAM_ROAD)

  while not do_exit:
    batch = {}
    for stream in order:
      buf = clients[stream].recv(100)
      if buf is None:
        break
      batch[stream] = buf
    if len(batch) != len(order):
      continue

    if start_frame_id is None:
      start_frame_id = max(clients[stream].frame_id for stream in order)

    frames_per_seg = SEGMENT_LENGTH * MAIN_FPS
    for stream in order:
      client = clients[stream]
      if client.frame_id < start_frame_id:
        continue
      if client.frame_id >= ((cur_seg[stream] + 1) * frames_per_seg) + start_frame_id:
        for e in encoders[stream]:
          e.encoder_close()
          e.encoder_open()
        cur_seg[stream] += 1

      extra = type("VisionIpcExtra", (), {
        "frame_id": client.frame_id,
        "timestamp_sof": client.timestamp_sof,
        "timestamp_eof": client.timestamp_eof,
      })()
      stream_encoders = encoders[stream]
      if stream == VisionStreamType.VISION_STREAM_ROAD and len(stream_encoders) > 1:
        stream_encoders = [stream_encoders[1], stream_encoders[0]]
      for e in stream_encoders:
        e.encode_frame(batch[stream], extra)


def _signal_handler(_signum, _frame) -> None:
  global do_exit
  do_exit = True


def main() -> None:
  signal.signal(signal.SIGINT, _signal_handler)
  signal.signal(signal.SIGTERM, _signal_handler)
  if not PC:
    config_realtime_process([0, 1, 2, 3], 52)
  import sys
  if len(sys.argv) > 1:
    if sys.argv[1] == "--stream":
      encoderd_thread(get_stream_cameras_logged())
    else:
      cloudlog.error("Argument '%s' is not supported", sys.argv[1])
  else:
    encoderd_thread(get_cameras_logged())


if __name__ == "__main__":
  main()
