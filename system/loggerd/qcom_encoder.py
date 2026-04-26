# ruff: noqa: F821
import ctypes
import errno
import fcntl
import os
import queue
import select
import threading

from cereal import log
from msgq.visionipc import VisionBuf
from openpilot.system.loggerd.c_header import load_c_constants
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.encoderd import EncoderInfo, VideoEncoder

ENCODE_TYPE = log.EncodeIndex.Type
VISIONBUF_SYNC_FROM_DEVICE = 0


class timeval(ctypes.Structure):
  _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_fract(ctypes.Structure):
  _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class v4l2_capability(ctypes.Structure):
  _fields_ = [("driver", ctypes.c_uint8 * 16), ("card", ctypes.c_uint8 * 32), ("bus_info", ctypes.c_uint8 * 32),
              ("version", ctypes.c_uint32), ("capabilities", ctypes.c_uint32), ("device_caps", ctypes.c_uint32),
              ("reserved", ctypes.c_uint32 * 3)]


class v4l2_plane_pix_format(ctypes.Structure):
  _fields_ = [("sizeimage", ctypes.c_uint32), ("bytesperline", ctypes.c_uint32), ("reserved", ctypes.c_uint16 * 6)]


class v4l2_pix_format_mplane(ctypes.Structure):
  _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("pixelformat", ctypes.c_uint32),
              ("field", ctypes.c_uint32), ("colorspace", ctypes.c_uint32),
              ("plane_fmt", v4l2_plane_pix_format * 8), ("num_planes", ctypes.c_uint8),
              ("flags", ctypes.c_uint8), ("ycbcr_enc", ctypes.c_uint8), ("quantization", ctypes.c_uint8),
              ("xfer_func", ctypes.c_uint8), ("reserved", ctypes.c_uint8 * 7)]


class v4l2_format_union(ctypes.Union):
  _fields_ = [("pix_mp", v4l2_pix_format_mplane), ("raw_data", ctypes.c_uint8 * 200)]


class v4l2_format(ctypes.Structure):
  _fields_ = [("type", ctypes.c_uint32), ("_padding", ctypes.c_uint32), ("fmt", v4l2_format_union)]


class v4l2_captureparm(ctypes.Structure):
  _fields_ = [("capability", ctypes.c_uint32), ("capturemode", ctypes.c_uint32), ("timeperframe", v4l2_fract),
              ("extendedmode", ctypes.c_uint32), ("readbuffers", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 4)]


class v4l2_outputparm(ctypes.Structure):
  _fields_ = [("capability", ctypes.c_uint32), ("outputmode", ctypes.c_uint32), ("timeperframe", v4l2_fract),
              ("extendedmode", ctypes.c_uint32), ("writebuffers", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 4)]


class v4l2_streamparm_union(ctypes.Union):
  _fields_ = [("capture", v4l2_captureparm), ("output", v4l2_outputparm), ("raw_data", ctypes.c_uint8 * 200)]


class v4l2_streamparm(ctypes.Structure):
  _fields_ = [("type", ctypes.c_uint32), ("parm", v4l2_streamparm_union)]


class v4l2_requestbuffers(ctypes.Structure):
  _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32), ("memory", ctypes.c_uint32),
              ("capabilities", ctypes.c_uint32), ("flags", ctypes.c_uint8), ("reserved", ctypes.c_uint8 * 3)]


class v4l2_plane_m_union(ctypes.Union):
  _fields_ = [("mem_offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong), ("fd", ctypes.c_int)]


class v4l2_plane(ctypes.Structure):
  _fields_ = [("bytesused", ctypes.c_uint32), ("length", ctypes.c_uint32), ("m", v4l2_plane_m_union),
              ("data_offset", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 11)]


class v4l2_buffer_m_union(ctypes.Union):
  _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong), ("planes", ctypes.POINTER(v4l2_plane)), ("fd", ctypes.c_int)]


class v4l2_timecode(ctypes.Structure):
  _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("frames", ctypes.c_uint8),
              ("seconds", ctypes.c_uint8), ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
              ("userbits", ctypes.c_uint8 * 4)]


class v4l2_buffer(ctypes.Structure):
  _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32), ("bytesused", ctypes.c_uint32),
              ("flags", ctypes.c_uint32), ("field", ctypes.c_uint32), ("timestamp", timeval),
              ("timecode", v4l2_timecode), ("sequence", ctypes.c_uint32), ("memory", ctypes.c_uint32),
              ("m", v4l2_buffer_m_union), ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
              ("request_fd", ctypes.c_int)]


class v4l2_control(ctypes.Structure):
  _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


class v4l2_encoder_cmd(ctypes.Structure):
  _fields_ = [("cmd", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("raw", ctypes.c_uint32 * 8)]


_C_CONSTANTS = [
  "VIDIOC_QUERYCAP",
  "VIDIOC_S_FMT",
  "VIDIOC_REQBUFS",
  "VIDIOC_QBUF",
  "VIDIOC_DQBUF",
  "VIDIOC_STREAMON",
  "VIDIOC_STREAMOFF",
  "VIDIOC_S_PARM",
  "VIDIOC_S_CTRL",
  "VIDIOC_ENCODER_CMD",
  "V4L2_BUF_FLAG_KEYFRAME",
  "V4L2_BUF_FLAG_TIMESTAMP_COPY",
  "V4L2_QCOM_BUF_FLAG_CODECCONFIG",
  "V4L2_QCOM_BUF_FLAG_EOS",
  "V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE",
  "V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE",
  "V4L2_MEMORY_USERPTR",
  "V4L2_FIELD_ANY",
  "V4L2_COLORSPACE_DEFAULT",
  "V4L2_COLORSPACE_470_SYSTEM_BG",
  "V4L2_PIX_FMT_HEVC",
  "V4L2_PIX_FMT_H264",
  "V4L2_PIX_FMT_NV12",
  "V4L2_ENC_CMD_STOP",
  "V4L2_CID_MPEG_VIDEO_BITRATE",
  "V4L2_CID_MPEG_VIDEO_HEADER_MODE",
  "V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE",
  "V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE",
  "V4L2_CID_MPEG_VIDEO_H264_LEVEL",
  "V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE",
  "V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA",
  "V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA",
  "V4L2_CID_MPEG_VIDEO_H264_PROFILE",
  "V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD",
  "V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES",
  "V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES",
  "V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME",
  "V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL",
  "V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL",
  "V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO",
  "V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE",
  "V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL",
  "V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY",
  "V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE",
  "V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR",
  "V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE",
  "V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN",
  "V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5",
  "V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED",
  "V4L2_MPEG_VIDEO_H264_PROFILE_HIGH",
  "V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN",
  "V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC",
  "V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0",
]
VIDIOC_QUERYCAP: int
VIDIOC_S_FMT: int
VIDIOC_REQBUFS: int
VIDIOC_QBUF: int
VIDIOC_DQBUF: int
VIDIOC_STREAMON: int
VIDIOC_STREAMOFF: int
VIDIOC_S_PARM: int
VIDIOC_S_CTRL: int
VIDIOC_ENCODER_CMD: int
V4L2_BUF_FLAG_KEYFRAME: int
V4L2_BUF_FLAG_TIMESTAMP_COPY: int
V4L2_QCOM_BUF_FLAG_CODECCONFIG: int
V4L2_QCOM_BUF_FLAG_EOS: int
V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE: int
V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE: int
V4L2_MEMORY_USERPTR: int
V4L2_FIELD_ANY: int
V4L2_COLORSPACE_DEFAULT: int
V4L2_COLORSPACE_470_SYSTEM_BG: int
V4L2_PIX_FMT_HEVC: int
V4L2_PIX_FMT_H264: int
V4L2_PIX_FMT_NV12: int
V4L2_ENC_CMD_STOP: int
V4L2_CID_MPEG_VIDEO_BITRATE: int
V4L2_CID_MPEG_VIDEO_HEADER_MODE: int
V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE: int
V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE: int
V4L2_CID_MPEG_VIDEO_H264_LEVEL: int
V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE: int
V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA: int
V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA: int
V4L2_CID_MPEG_VIDEO_H264_PROFILE: int
V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD: int
V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES: int
V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES: int
V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME: int
V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL: int
V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL: int
V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO: int
V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE: int
V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL: int
V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY: int
V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE: int
V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR: int
V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE: int
V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN: int
V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5: int
V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED: int
V4L2_MPEG_VIDEO_H264_PROFILE_HIGH: int
V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN: int
V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC: int
V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0: int
globals().update(load_c_constants([
  "system/loggerd/qcom_v4l2.h",
  "third_party/linux/include/v4l2-controls.h",
  "<linux/videodev2.h>",
], _C_CONSTANTS))


def safe_ioctl(fd: int, request: int, arg, name: str) -> None:
  while True:
    try:
      fcntl.ioctl(fd, request, arg)
      return
    except OSError as e:
      if e.errno == errno.EINTR:
        continue
      raise RuntimeError(f"{name}: errno={e.errno}") from e


class V4LEncoder(VideoEncoder):
  BUF_IN_COUNT = 7
  BUF_OUT_COUNT = 6

  def __init__(self, encoder_info: EncoderInfo, in_width: int, in_height: int) -> None:
    super().__init__(encoder_info, in_width, in_height)
    self.fd = os.open("/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1", os.O_RDWR | os.O_NONBLOCK)
    cap = v4l2_capability()
    safe_ioctl(self.fd, VIDIOC_QUERYCAP, cap, "VIDIOC_QUERYCAP failed")
    cloudlog.debug("opened encoder device %s %s = %d", bytes(cap.driver).split(b"\0")[0], bytes(cap.card).split(b"\0")[0], self.fd)

    settings = encoder_info.get_settings(in_width)
    is_h265 = settings.encode_type == ENCODE_TYPE.fullHEVC

    fmt_out = v4l2_format()
    fmt_out.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
    fmt_out.fmt.pix_mp.width = self.out_width
    fmt_out.fmt.pix_mp.height = self.out_height
    fmt_out.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_HEVC if is_h265 else V4L2_PIX_FMT_H264
    fmt_out.fmt.pix_mp.field = V4L2_FIELD_ANY
    fmt_out.fmt.pix_mp.colorspace = V4L2_COLORSPACE_DEFAULT
    safe_ioctl(self.fd, VIDIOC_S_FMT, fmt_out, "VIDIOC_S_FMT capture failed")

    streamparm = v4l2_streamparm()
    streamparm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    streamparm.parm.output.timeperframe.numerator = 1
    streamparm.parm.output.timeperframe.denominator = encoder_info.fps
    safe_ioctl(self.fd, VIDIOC_S_PARM, streamparm, "VIDIOC_S_PARM failed")

    fmt_in = v4l2_format()
    fmt_in.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    fmt_in.fmt.pix_mp.width = in_width
    fmt_in.fmt.pix_mp.height = in_height
    fmt_in.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12
    fmt_in.fmt.pix_mp.field = V4L2_FIELD_ANY
    fmt_in.fmt.pix_mp.colorspace = V4L2_COLORSPACE_470_SYSTEM_BG
    safe_ioctl(self.fd, VIDIOC_S_FMT, fmt_in, "VIDIOC_S_FMT output failed")

    shared_ctrls = (
      (V4L2_CID_MPEG_VIDEO_BITRATE, settings.bitrate),
      (V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, settings.gop_size - settings.b_frames - 1),
      (V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, settings.b_frames),
      (V4L2_CID_MPEG_VIDEO_HEADER_MODE, V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE),
      (V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL, V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR),
      (V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE),
      (V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, 1),
    )
    codec_ctrls = (
      ((V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE, V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN),
       (V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5),
       (V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO, V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED))
      if is_h265 else
      ((V4L2_CID_MPEG_VIDEO_H264_PROFILE, V4L2_MPEG_VIDEO_H264_PROFILE_HIGH),
       (V4L2_CID_MPEG_VIDEO_H264_LEVEL, V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN),
       (V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC),
       (V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0),
       (V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, 0), (V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA, 0),
       (V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA, 0), (V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE, 0))
    )
    for ctrl_id, value in (*shared_ctrls, *codec_ctrls):
      ctrl = v4l2_control(ctrl_id, value)
      safe_ioctl(self.fd, VIDIOC_S_CTRL, ctrl, f"VIDIOC_S_CTRL failed id=0x{ctrl_id:x} value={value}")

    self._request_buffers(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, self.BUF_OUT_COUNT)
    self._request_buffers(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, self.BUF_IN_COUNT)
    self._stream_on(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
    self._stream_on(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)

    out_size = fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage
    self.buf_out = []
    for i in range(self.BUF_OUT_COUNT):
      b = VisionBuf()
      b.allocate(out_size)
      self.buf_out.append(b)
      self._queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, b)
    self.free_buf_in: queue.Queue[int] = queue.Queue()
    for i in range(self.BUF_IN_COUNT):
      self.free_buf_in.put(i)
    self.extras: queue.Queue = queue.Queue()
    self.segment_num = -1
    self.counter = 0
    self.segment_id = -1
    self.rotate_pending = False
    self.segment_lock = threading.Lock()
    self.is_open = False
    self.dequeue_thread: threading.Thread | None = None

  def _request_buffers(self, buf_type: int, count: int) -> None:
    req = v4l2_requestbuffers(count=count, type=buf_type, memory=V4L2_MEMORY_USERPTR)
    safe_ioctl(self.fd, VIDIOC_REQBUFS, req, "VIDIOC_REQBUFS failed")

  def _stream_on(self, buf_type: int) -> None:
    typ = ctypes.c_int(buf_type)
    safe_ioctl(self.fd, VIDIOC_STREAMON, typ, "VIDIOC_STREAMON failed")

  def _stream_off(self, buf_type: int) -> None:
    typ = ctypes.c_int(buf_type)
    safe_ioctl(self.fd, VIDIOC_STREAMOFF, typ, "VIDIOC_STREAMOFF failed")

  def _dequeue_buffer(self, buf_type: int):
    plane = v4l2_plane()
    vbuf = v4l2_buffer()
    vbuf.type = buf_type
    vbuf.memory = V4L2_MEMORY_USERPTR
    vbuf.m.planes = ctypes.pointer(plane)
    vbuf.length = 1
    safe_ioctl(self.fd, VIDIOC_DQBUF, vbuf, "VIDIOC_DQBUF failed")
    assert plane.data_offset == 0
    return vbuf.index, plane.bytesused, vbuf.flags, vbuf.timestamp

  def _queue_buffer(self, buf_type: int, index: int, buf: VisionBuf, timestamp: timeval | None = None) -> None:
    plane = v4l2_plane()
    plane.bytesused = buf.buffer_len if hasattr(buf, "buffer_len") else len(buf.data)
    plane.length = plane.bytesused
    plane.m.userptr = buf.addr
    plane.reserved[0] = buf.fd
    vbuf = v4l2_buffer()
    vbuf.index = index
    vbuf.type = buf_type
    vbuf.flags = V4L2_BUF_FLAG_TIMESTAMP_COPY
    if timestamp is not None:
      vbuf.timestamp = timestamp
    vbuf.memory = V4L2_MEMORY_USERPTR
    vbuf.m.planes = ctypes.pointer(plane)
    vbuf.length = 1
    safe_ioctl(self.fd, VIDIOC_QBUF, vbuf, "VIDIOC_QBUF failed")

  def encoder_open(self) -> None:
    self.segment_num += 1
    self.counter = 0
    self.segment_id = -1
    self.rotate_pending = False
    self.dequeue_thread = threading.Thread(target=self._dequeue_handler, name=f"dq-{self.encoder_info.publish_name}")
    self.dequeue_thread.daemon = True
    self.dequeue_thread.start()
    self.is_open = True

  def encoder_rotate(self) -> None:
    with self.segment_lock:
      self.rotate_pending = True
    ctrl = v4l2_control(V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME, 1)
    safe_ioctl(self.fd, VIDIOC_S_CTRL, ctrl, "VIDIOC_S_CTRL request iframe failed")

  def _dequeue_handler(self) -> None:
    idx = -1
    header = b""
    poller = select.poll()
    poller.register(self.fd, select.POLLIN | select.POLLOUT)
    while True:
      events = poller.poll(1000)
      if not events:
        cloudlog.error("encoder dequeue poll timeout")
        continue
      revents = events[0][1]
      if revents & select.POLLIN:
        index, bytesused, flags, ts = self._dequeue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        self.buf_out[index].sync(VISIONBUF_SYNC_FROM_DEVICE)
        data = bytes(memoryview(self.buf_out[index].data)[:bytesused])
        timestamp_us = ts.tv_sec * 1_000_000 + ts.tv_usec
        if flags & V4L2_QCOM_BUF_FLAG_EOS:
          self._queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, self.buf_out[index])
          break
        if flags & V4L2_QCOM_BUF_FLAG_CODECCONFIG:
          header = data
        else:
          extra = self.extras.get()
          assert extra.timestamp_eof // 1000 == timestamp_us
          with self.segment_lock:
            if self.rotate_pending and (flags & V4L2_BUF_FLAG_KEYFRAME):
              self.segment_num += 1
              self.segment_id = -1
              self.rotate_pending = False
            segment_num = self.segment_num
            self.segment_id += 1
            idx = self.segment_id
          self.publisher_publish(segment_num, idx, extra, flags, header, data)
        self._queue_buffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, self.buf_out[index])
      if revents & select.POLLOUT:
        index, _bytesused, _flags, _ts = self._dequeue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
        self.free_buf_in.put(index)

  def encode_frame(self, buf: VisionBuf, extra) -> int:
    timestamp = timeval(extra.timestamp_eof // 1_000_000_000, (extra.timestamp_eof // 1000) % 1_000_000)
    buffer_in = self.free_buf_in.get()
    self.extras.put(extra)
    self._queue_buffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, buf, timestamp)
    ret = self.counter
    self.counter += 1
    return ret

  def encoder_close(self) -> None:
    if not self.is_open:
      return
    for _ in range(self.BUF_IN_COUNT):
      self.free_buf_in.get()
    for i in range(self.BUF_IN_COUNT):
      self.free_buf_in.put(i)
    cmd = v4l2_encoder_cmd()
    cmd.cmd = V4L2_ENC_CMD_STOP
    safe_ioctl(self.fd, VIDIOC_ENCODER_CMD, cmd, "VIDIOC_ENCODER_CMD failed")
    assert self.dequeue_thread is not None
    self.dequeue_thread.join()
    assert self.extras.empty()
    self.is_open = False

  def close(self) -> None:
    self.encoder_close()
    self._stream_off(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
    self._request_buffers(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, 0)
    self._stream_off(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
    self._request_buffers(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, 0)
    os.close(self.fd)
    for b in self.buf_out:
      if b.free() != 0:
        cloudlog.error("Failed to free buffer")
