import time
import threading
import os
import fcntl
import ctypes as C
import select
import mmap
import numpy as np

import cereal.messaging as messaging

from openpilot.system.loggerd.encoder.encoder import VideoEncoder
from openpilot.system.loggerd.encoder.v4l2_ctypes import (
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
  VIDIOC_STREAMOFF, VIDIOC_ENCODER_CMD, v4l2_encoder_cmd, V4L2_ENC_CMD_STOP,
  V4L2_BUF_FLAG_TIMESTAMP_COPY, timeval,
)


class V4LEncoder(VideoEncoder):
  def open(self) -> None:

    dev = "/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1"
    self._fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)

    cap = v4l2_capability()
    fcntl.ioctl(self._fd, VIDIOC_QUERYCAP, cap)

    settings = self.encoder_info.get_settings(self.in_width)
    is_h265 = settings.encode_type in ("FULL_H_E_V_C",)
    out_pix = V4L2_PIX_FMT_HEVC if is_h265 else V4L2_PIX_FMT_H264

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

    sparm = v4l2_streamparm()
    sparm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    sparm.parm = v4l2_streamparm_parm()
    op = v4l2_outputparm()
    op.timeperframe = v4l2_fract(1, self.encoder_info.fps)
    sparm.parm.output = op
    fcntl.ioctl(self._fd, VIDIOC_S_PARM, sparm)

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

    req_cap = v4l2_requestbuffers()
    req_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
    req_cap.memory = V4L2_MEMORY_MMAP
    req_cap.count = 6
    fcntl.ioctl(self._fd, VIDIOC_REQBUFS, req_cap)

    req_out = v4l2_requestbuffers()
    req_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
    req_out.memory = V4L2_MEMORY_USERPTR
    req_out.count = 7
    fcntl.ioctl(self._fd, VIDIOC_REQBUFS, req_out)

    # map & queue capture buffers
    self._cap_buf_count = req_cap.count
    self._cap_mmaps = []
    self._cap_lengths = []
    for i in range(self._cap_buf_count):
      plane = v4l2_plane()
      buf = v4l2_buffer()
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
      buf.index = i
      buf.memory = V4L2_MEMORY_MMAP
      buf.length = 1
      buf.m = v4l2_buffer_m()
      buf.m.planes = C.pointer(plane)
      fcntl.ioctl(self._fd, VIDIOC_QUERYBUF, buf)
      length = plane.length
      offset = plane.m.mem_offset
      mm = mmap.mmap(self._fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
      self._cap_mmaps.append(mm)
      self._cap_lengths.append(length)

      plane_q = v4l2_plane()
      plane_q.length = length
      qbuf = v4l2_buffer()
      qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
      qbuf.index = i
      qbuf.memory = V4L2_MEMORY_MMAP
      qbuf.length = 1
      qbuf.m = v4l2_buffer_m()
      qbuf.m.planes = C.pointer(plane_q)
      fcntl.ioctl(self._fd, VIDIOC_QBUF, qbuf)

    fcntl.ioctl(self._fd, VIDIOC_STREAMON, C.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE))
    fcntl.ioctl(self._fd, VIDIOC_STREAMON, C.c_uint32(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE))

    self.segment_num += 1
    self.counter = 0
    self.is_open = True
    self._free_in = list(range(req_out.count))
    self._free_lock = threading.Lock()
    self._free_cv = threading.Condition(self._free_lock)
    self._extras = []  # (frame_id, ts_sof, ts_eof)
    self._extras_lock = threading.Lock()
    self._header = None

    def dq_thread():
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
          continue
        for _fd, revents in events:
          if revents & POLLIN:
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
            payload = self._cap_mmaps[idx][:bytesused]
            if flags & V4L2_QCOM_BUF_FLAG_EOS:
              self.is_open = False
            elif flags & V4L2_QCOM_BUF_FLAG_CODECCONFIG:
              self._header = bytes(payload)
            else:
              with self._extras_lock:
                if self._extras:
                  frame_id, ts_sof, ts_eof = self._extras.pop(0)
                else:
                  frame_id, ts_sof, ts_eof = 0, 0, int(time.time_ns())
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
    if getattr(self, "is_open", False):
      try:
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

  def encode_frame(self, buf, frame_id: int, timestamp_sof: int, timestamp_eof: int) -> int:
    with self._free_cv:
      while not self._free_in:
        self._free_cv.wait(timeout=0.01)
        if not self.is_open:
          return -1
      index = self._free_in.pop(0)

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

    with self._extras_lock:
      self._extras.append((frame_id, timestamp_sof, timestamp_eof))

    try:
      fcntl.ioctl(self._fd, VIDIOC_QBUF, vbuf)
    except Exception:
      return -1
    return self.counter
