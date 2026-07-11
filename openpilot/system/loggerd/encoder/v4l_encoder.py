import ctypes
import os
import queue
import select
import threading
import time

from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.encoder.encoder import DEBUG_ENCODER, EncoderInfo, FrameExtra, VideoEncoder
from openpilot.system.loggerd.encoder.ion import IonBuf
from openpilot.system.loggerd.encoder import v4l2
from openpilot.system.loggerd.encoder.v4l2 import safe_ioctl, v4l2_buffer, v4l2_control, v4l2_plane

from openpilot.cereal import log

VIDEO_DEVICE = "/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1"

BUF_IN_COUNT = 7
BUF_OUT_COUNT = 6

# kernel debugging:
# echo 0xff > /sys/module/videobuf2_core/parameters/debug
# echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level
# echo 0xff > /sys/devices/platform/soc/aa00000.qcom,vidc/video4linux/video33/dev_debug


def millis_since_boot() -> float:
  return time.clock_gettime(time.CLOCK_BOOTTIME) * 1000.


def dequeue_buffer(fd: int, buf_type: int) -> tuple[int, int, int, int]:
  plane = v4l2_plane()
  v4l_buf = v4l2_buffer(type=buf_type, memory=v4l2.V4L2_MEMORY_USERPTR, length=1)
  v4l_buf.m.planes = ctypes.pointer(plane)
  safe_ioctl(fd, v4l2.VIDIOC_DQBUF, v4l_buf)

  assert plane.data_offset == 0
  ts = v4l_buf.timestamp.tv_sec * 1000000 + v4l_buf.timestamp.tv_usec
  return v4l_buf.index, plane.bytesused, v4l_buf.flags, ts


def queue_buffer(fd: int, buf_type: int, index: int, addr: int, length: int, buf_fd: int, timestamp_eof: int = 0) -> None:
  plane = v4l2_plane(bytesused=length, length=length)
  plane.m.userptr = addr
  plane.reserved[0] = buf_fd

  v4l_buf = v4l2_buffer(index=index, type=buf_type, flags=v4l2.V4L2_BUF_FLAG_TIMESTAMP_COPY,
                        memory=v4l2.V4L2_MEMORY_USERPTR, length=1)
  v4l_buf.timestamp.tv_sec = timestamp_eof // 1_000_000_000
  v4l_buf.timestamp.tv_usec = (timestamp_eof // 1000) % 1_000_000
  v4l_buf.m.planes = ctypes.pointer(plane)
  safe_ioctl(fd, v4l2.VIDIOC_QBUF, v4l_buf)


def request_buffers(fd: int, buf_type: int, count: int) -> None:
  reqbuf = v4l2.v4l2_requestbuffers(count=count, type=buf_type, memory=v4l2.V4L2_MEMORY_USERPTR)
  safe_ioctl(fd, v4l2.VIDIOC_REQBUFS, reqbuf)


class V4LEncoder(VideoEncoder):
  def __init__(self, encoder_info: EncoderInfo, in_width: int, in_height: int):
    super().__init__(encoder_info, in_width, in_height)

    self.fd = os.open(VIDEO_DEVICE, os.O_RDWR | os.O_NONBLOCK)

    cap = v4l2.v4l2_capability()
    safe_ioctl(self.fd, v4l2.VIDIOC_QUERYCAP, cap)
    cloudlog.debug(f"opened encoder device {cap.driver} {cap.card} = {self.fd}")
    assert cap.driver == b"msm_vidc_driver"
    assert cap.card == b"msm_vidc_venc"

    encoder_settings = encoder_info.get_settings(in_width)
    self.current_bitrate = encoder_settings.bitrate
    is_h265 = encoder_settings.encode_type == log.EncodeIndex.Type.fullHEVC

    fmt_out = v4l2.v4l2_format(type=v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
    fmt_out.fmt.pix_mp.width = self.out_width  # downscales are free with v4l
    fmt_out.fmt.pix_mp.height = self.out_height
    fmt_out.fmt.pix_mp.pixelformat = v4l2.V4L2_PIX_FMT_HEVC if is_h265 else v4l2.V4L2_PIX_FMT_H264
    fmt_out.fmt.pix_mp.field = v4l2.V4L2_FIELD_ANY
    fmt_out.fmt.pix_mp.colorspace = v4l2.V4L2_COLORSPACE_DEFAULT
    safe_ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt_out)

    streamparm = v4l2.v4l2_streamparm(type=v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
    streamparm.parm.output.timeperframe.numerator = 1
    streamparm.parm.output.timeperframe.denominator = encoder_info.fps
    safe_ioctl(self.fd, v4l2.VIDIOC_S_PARM, streamparm)

    fmt_in = v4l2.v4l2_format(type=v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
    fmt_in.fmt.pix_mp.width = in_width
    fmt_in.fmt.pix_mp.height = in_height
    fmt_in.fmt.pix_mp.pixelformat = v4l2.V4L2_PIX_FMT_NV12
    fmt_in.fmt.pix_mp.field = v4l2.V4L2_FIELD_ANY
    fmt_in.fmt.pix_mp.colorspace = v4l2.V4L2_COLORSPACE_470_SYSTEM_BG
    safe_ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt_in)

    cloudlog.debug(f"in buffer size {fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage}, out buffer size {fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage}")

    # shared ctrls
    ctrls = [
      (v4l2.V4L2_CID_MPEG_VIDEO_BITRATE, encoder_settings.bitrate),
      (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, encoder_settings.gop_size - encoder_settings.b_frames - 1),
      (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, encoder_settings.b_frames),
      (v4l2.V4L2_CID_MPEG_VIDEO_HEADER_MODE, v4l2.V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE),
      (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL, v4l2.V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR),
      (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, v4l2.V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE),
      (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, 1),
    ]
    if is_h265:
      ctrls += [
        (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE, v4l2.V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN),
        (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, v4l2.V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5),
        (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO, v4l2.V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED),
      ]
    else:
      ctrls += [
        (v4l2.V4L2_CID_MPEG_VIDEO_H264_PROFILE, v4l2.V4L2_MPEG_VIDEO_H264_PROFILE_HIGH),
        (v4l2.V4L2_CID_MPEG_VIDEO_H264_LEVEL, v4l2.V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN),
        (v4l2.V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, v4l2.V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC),
        (v4l2.V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, v4l2.V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0),
        (v4l2.V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, 0),
        (v4l2.V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA, 0),
        (v4l2.V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA, 0),
        (v4l2.V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE, 0),
      ]
    for ctrl_id, value in ctrls:
      safe_ioctl(self.fd, v4l2.VIDIOC_S_CTRL, v4l2_control(id=ctrl_id, value=value))

    # allocate buffers
    request_buffers(self.fd, v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, BUF_OUT_COUNT)
    request_buffers(self.fd, v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, BUF_IN_COUNT)

    # start encoder
    for buf_type in (v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE):
      safe_ioctl(self.fd, v4l2.VIDIOC_STREAMON, ctypes.c_int(buf_type))

    # queue up output buffers
    self.buf_out = []
    for i in range(BUF_OUT_COUNT):
      buf = IonBuf(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage)
      self.buf_out.append(buf)
      queue_buffer(self.fd, v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, buf.addr, buf.len, buf.fd)
    # queue up input buffers
    self.free_buf_in: queue.Queue[int] = queue.Queue()
    for i in range(BUF_IN_COUNT):
      self.free_buf_in.put(i)

    self.extras: queue.Queue[FrameExtra] = queue.Queue()
    self.is_open = False
    self.segment_num = -1
    self.counter = 0

  def _dequeue_handler(self) -> None:
    self.segment_num += 1
    idx = -1
    exit_flag = False

    # POLLIN is capture, POLLOUT is frame
    poller = select.poll()
    poller.register(self.fd, select.POLLIN | select.POLLOUT)

    # save the header
    header = b""

    while not exit_flag:
      events = poller.poll(1000)
      if not events:
        cloudlog.error("encoder dequeue poll timeout")
        continue
      revents = events[0][1]

      if DEBUG_ENCODER >= 2:
        print(f"{self.encoder_info.publish_name:>20} poll {revents:x} at {millis_since_boot():.2f} ms")

      frame_id = -1
      if revents & select.POLLIN:
        index, bytesused, flags, ts = dequeue_buffer(self.fd, v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        buf = self.buf_out[index]
        buf.sync_from_device()

        if flags & v4l2.V4L2_QCOM_BUF_FLAG_EOS:
          # eof packet, we exit
          exit_flag = True
        elif flags & v4l2.V4L2_QCOM_BUF_FLAG_CODECCONFIG:
          header = buf.mm[:bytesused]
        else:
          extra = self.extras.get()
          assert extra.timestamp_eof // 1000 == ts  # stay in sync
          frame_id = extra.frame_id
          idx += 1
          self.publisher_publish(self.segment_num, idx, extra, flags, header, buf.mm[:bytesused])

        if DEBUG_ENCODER:
          lat = millis_since_boot() - ts / 1000.
          print(f"{self.encoder_info.publish_name:>20} got({index}) {bytesused:6d} bytes flags {flags:8x} " +
                f"idx {self.segment_num:3d}/{idx:4d} id {frame_id:8d} ts {ts} lat {lat:.2f} ms ({self.free_buf_in.qsize()} frames free)")

        # requeue the buffer
        queue_buffer(self.fd, v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, buf.addr, buf.len, buf.fd)

      if revents & select.POLLOUT:
        index, _, _, _ = dequeue_buffer(self.fd, v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
        self.free_buf_in.put(index)

  def encoder_open(self) -> None:
    self.dequeue_handler_thread = threading.Thread(target=self._dequeue_handler, name=f"dq-{self.encoder_info.publish_name}")
    self.dequeue_handler_thread.start()
    self.is_open = True
    self.counter = 0

  def encode_frame(self, buf, extra: FrameExtra) -> int:
    # reserve buffer
    buffer_in = self.free_buf_in.get()

    # push buffer
    self.extras.put(extra)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf.data))
    queue_buffer(self.fd, v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buffer_in, addr, len(buf.data), buf.fd, extra.timestamp_eof)

    self.counter += 1
    return self.counter - 1

  def encoder_close(self) -> None:
    if self.is_open:
      # pop all the frames before closing, then put the buffers back
      for _ in range(BUF_IN_COUNT):
        self.free_buf_in.get()
      for i in range(BUF_IN_COUNT):
        self.free_buf_in.put(i)
      # no frames, stop the encoder
      encoder_cmd = v4l2.v4l2_encoder_cmd(cmd=v4l2.V4L2_ENC_CMD_STOP)
      safe_ioctl(self.fd, v4l2.VIDIOC_ENCODER_CMD, encoder_cmd)
      # join waits for V4L2_QCOM_BUF_FLAG_EOS
      self.dequeue_handler_thread.join()
      assert self.extras.empty()
    self.is_open = False

  def set_bitrate(self, bitrate: int) -> None:
    if bitrate == self.current_bitrate:
      return
    if bitrate <= 0:
      cloudlog.error(f"invalid livestream encoder bitrate {bitrate}")
      return

    try:
      safe_ioctl(self.fd, v4l2.VIDIOC_S_CTRL, v4l2_control(id=v4l2.V4L2_CID_MPEG_VIDEO_BITRATE, value=bitrate))
    except OSError:
      cloudlog.error(f"failed to update {self.encoder_info.publish_name} bitrate to {bitrate}")
      return
    self.current_bitrate = bitrate

  def request_keyframe(self) -> None:
    try:
      safe_ioctl(self.fd, v4l2.VIDIOC_S_CTRL, v4l2_control(id=v4l2.V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME, value=1))
    except OSError:
      cloudlog.error(f"failed to request keyframe for {self.encoder_info.publish_name}")

  def close(self) -> None:
    self.encoder_close()
    for buf_type in (v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE):
      safe_ioctl(self.fd, v4l2.VIDIOC_STREAMOFF, ctypes.c_int(buf_type))
      request_buffers(self.fd, buf_type, 0)
    os.close(self.fd)

    for buf in self.buf_out:
      try:
        buf.free()
      except OSError:
        cloudlog.error("Failed to free buffer")
