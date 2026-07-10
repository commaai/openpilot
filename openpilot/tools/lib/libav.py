import ctypes
import errno
import os
from functools import cache
from types import SimpleNamespace

import numpy as np

AV_LOG_ERROR = 16
AV_INPUT_BUFFER_PADDING_SIZE = 64  # required padding after packet data, must be zeroed

AVERROR_EAGAIN = -errno.EAGAIN
AVERROR_EOF = -int.from_bytes(b"EOF ", "little")  # FFERRTAG('E', 'O', 'F', ' ')

# each FFmpeg release bumps these majors in lockstep: FFmpeg 4.x is avcodec 58/avutil 56, ..., FFmpeg 8.x is 62/60
SUPPORTED_MAJORS = {"avcodec": range(58, 63), "avutil": range(56, 61)}


class AVFrame(ctypes.Structure):
  # leading fields only, the rest of the struct is never accessed
  _fields_ = [
    ("data", ctypes.c_void_p * 8),
    ("linesize", ctypes.c_int * 8),
    ("extended_data", ctypes.c_void_p),
    ("width", ctypes.c_int),
    ("height", ctypes.c_int),
    ("nb_samples", ctypes.c_int),
    ("format", ctypes.c_int),
  ]


_FUNCTIONS = {
  "avcodec": {
    "avcodec_find_decoder_by_name": (ctypes.c_void_p, [ctypes.c_char_p]),
    "avcodec_alloc_context3": (ctypes.c_void_p, [ctypes.c_void_p]),
    "avcodec_free_context": (None, [ctypes.POINTER(ctypes.c_void_p)]),
    "avcodec_open2": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]),
    "avcodec_send_packet": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
    "avcodec_receive_frame": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(AVFrame)]),
    "av_packet_alloc": (ctypes.c_void_p, []),
    "av_packet_free": (None, [ctypes.POINTER(ctypes.c_void_p)]),
    "av_packet_from_data": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
  },
  "avutil": {
    "av_frame_alloc": (ctypes.POINTER(AVFrame), []),
    "av_frame_free": (None, [ctypes.POINTER(ctypes.POINTER(AVFrame))]),
    "av_frame_unref": (None, [ctypes.POINTER(AVFrame)]),
    "av_malloc": (ctypes.c_void_p, [ctypes.c_size_t]),
    "av_free": (None, [ctypes.c_void_p]),
    "av_strerror": (ctypes.c_int, [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]),
    "av_opt_set": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]),
    "av_log_set_level": (None, [ctypes.c_int]),
    "av_get_pix_fmt": (ctypes.c_int, [ctypes.c_char_p]),
  },
}


@cache
def _libav() -> SimpleNamespace:
  from ffmpeg import LIB_DIR

  # unversioned .so files are ld scripts; load the real soname next to them
  libs = {}
  for name in _FUNCTIONS:
    path = next(os.path.join(LIB_DIR, f) for f in os.listdir(LIB_DIR) if f.startswith(f"lib{name}.so."))
    libs[name] = ctypes.CDLL(path)

  for name, lib in libs.items():
    version_fn = getattr(lib, f"{name}_version")
    version_fn.restype = ctypes.c_uint
    major = version_fn() >> 16
    if major not in SUPPORTED_MAJORS[name]:
      raise RuntimeError(f"lib{name} major version {major} is unsupported, extend SUPPORTED_MAJORS after checking the AVFrame layout still matches")

  av = SimpleNamespace()
  for name, lib in libs.items():
    for func, (restype, argtypes) in _FUNCTIONS[name].items():
      f = getattr(lib, func)
      f.restype, f.argtypes = restype, argtypes
      setattr(av, func, f)

  # look up pixel format enum values by name instead of baking in constants
  av.AV_PIX_FMT_YUV420P = av.av_get_pix_fmt(b"yuv420p")
  av.AV_PIX_FMT_YUVJ420P = av.av_get_pix_fmt(b"yuvj420p")
  av.AV_PIX_FMT_NV12 = av.av_get_pix_fmt(b"nv12")

  av.av_log_set_level(AV_LOG_ERROR)  # decoders are chatty about corrupt/partial input
  return av


def _plane(frame: AVFrame, idx: int, rows: int, cols: int) -> np.ndarray:
  # a plane is `rows` lines of `cols` pixels, each padded out to linesize bytes.
  # FFmpeg pads its image allocations, so viewing the padding of the last line is safe.
  stride = frame.linesize[idx]
  if not frame.data[idx] or stride < cols:
    raise RuntimeError(f"decoded frame has invalid plane {idx}")
  buf = (ctypes.c_uint8 * (stride * rows)).from_address(frame.data[idx])
  return np.frombuffer(buf, dtype=np.uint8).reshape(rows, stride)[:, :cols]


class VideoDecoder:
  """Streaming decoder: feed compressed packets, get back decoded NV12 frames."""

  def __init__(self, codec: str = "hevc"):
    self._ctx = None
    self._frame = None
    self._av = _libav()

    decoder = self._av.avcodec_find_decoder_by_name(codec.encode())
    if not decoder:
      raise ValueError(f"no FFmpeg decoder found for codec {codec!r}")

    self._ctx = ctypes.c_void_p(self._av.avcodec_alloc_context3(decoder))
    self._frame = self._av.av_frame_alloc()
    if not self._ctx or not self._frame:
      raise MemoryError("failed to allocate decoder state")

    # output frames as soon as they decode instead of buffering for reordering,
    # and only allow slice threading, which doesn't add any frame delay
    self._check(self._av.av_opt_set(self._ctx, b"flags", b"+low_delay", 0), "av_opt_set flags")
    self._check(self._av.av_opt_set(self._ctx, b"thread_type", b"slice", 0), "av_opt_set thread_type")
    self._check(self._av.avcodec_open2(self._ctx, decoder, None), f"failed to open {codec!r} decoder")

  def decode(self, data: bytes | bytearray | memoryview) -> list[np.ndarray]:
    """Decode one compressed packet, returning any completed frames in display order.

    Frames are (height * 3 // 2, width) NV12 arrays: the luma plane on top of the
    interleaved chroma plane.
    """
    data = bytes(data)
    if len(data) == 0:
      return []

    frames = []
    pkt = ctypes.c_void_p(self._av.av_packet_alloc())
    if not pkt:
      raise MemoryError("failed to allocate packet")
    try:
      buf = self._av.av_malloc(len(data) + AV_INPUT_BUFFER_PADDING_SIZE)
      if not buf:
        raise MemoryError("failed to allocate packet buffer")
      ctypes.memmove(buf, data, len(data))
      ctypes.memset(buf + len(data), 0, AV_INPUT_BUFFER_PADDING_SIZE)
      ret = self._av.av_packet_from_data(pkt, buf, len(data))
      if ret < 0:
        self._av.av_free(buf)
        self._check(ret, "av_packet_from_data")

      ret = self._av.avcodec_send_packet(self._ctx, pkt)
      if ret == AVERROR_EAGAIN:  # the decoder wants to be drained before taking more input
        frames.extend(self._receive_frames())
        ret = self._av.avcodec_send_packet(self._ctx, pkt)
      self._check(ret, "avcodec_send_packet")
    finally:
      self._av.av_packet_free(ctypes.byref(pkt))

    frames.extend(self._receive_frames())
    return frames

  def close(self) -> None:
    if self._frame:
      self._av.av_frame_free(ctypes.byref(self._frame))
    if self._ctx:
      self._av.avcodec_free_context(ctypes.byref(self._ctx))

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()

  def __del__(self):
    if getattr(self, "_av", None) is not None:
      self.close()

  def _check(self, ret: int, msg: str) -> None:
    if ret < 0:
      errbuf = ctypes.create_string_buffer(256)
      self._av.av_strerror(ret, errbuf, len(errbuf))
      raise RuntimeError(f"{msg}: {errbuf.value.decode(errors='replace')}")

  def _receive_frames(self) -> list[np.ndarray]:
    frames = []
    while True:
      ret = self._av.avcodec_receive_frame(self._ctx, self._frame)
      if ret in (AVERROR_EAGAIN, AVERROR_EOF):
        return frames
      self._check(ret, "avcodec_receive_frame")
      try:
        frames.append(self._frame_to_nv12(self._frame.contents))
      finally:
        self._av.av_frame_unref(self._frame)

  def _frame_to_nv12(self, frame: AVFrame) -> np.ndarray:
    w, h = frame.width, frame.height
    if w % 2 or h % 2:
      raise RuntimeError(f"NV12 output requires even frame dimensions, got {w}x{h}")

    nv12 = np.empty((h * 3 // 2, w), dtype=np.uint8)
    y, uv = nv12[:h], nv12[h:]
    y[:] = _plane(frame, 0, h, w)
    if frame.format == self._av.AV_PIX_FMT_NV12:
      uv[:] = _plane(frame, 1, h // 2, w)
    elif frame.format in (self._av.AV_PIX_FMT_YUV420P, self._av.AV_PIX_FMT_YUVJ420P):
      uv[:, 0::2] = _plane(frame, 1, h // 2, w // 2)
      uv[:, 1::2] = _plane(frame, 2, h // 2, w // 2)
    else:
      raise RuntimeError(f"unsupported decoded pixel format {frame.format}")
    return nv12
