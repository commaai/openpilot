import ctypes
import errno
import glob
import os
from collections.abc import Generator

import ffmpeg
import numpy as np


AVERROR_EOF = -541478725
AV_INPUT_BUFFER_PADDING_SIZE = 64
AV_LOG_QUIET = -8
AV_PIX_FMT_NV12 = 23
SWS_FAST_BILINEAR = 1


class FFmpegError(RuntimeError):
  pass


class AVPacket(ctypes.Structure):
  # Public prefix of AVPacket. Only data and size are modified here; the packet
  # remains non-refcounted and points at Decoder._packet_buffer.
  _fields_ = [
    ("buf", ctypes.c_void_p),
    ("pts", ctypes.c_int64),
    ("dts", ctypes.c_int64),
    ("data", ctypes.POINTER(ctypes.c_uint8)),
    ("size", ctypes.c_int),
  ]


class AVFrame(ctypes.Structure):
  # Public prefix of AVFrame through format. FFmpeg keeps these fields ABI
  # stable; keeping the declaration short avoids depending on private fields.
  _fields_ = [
    ("data", ctypes.POINTER(ctypes.c_uint8) * 8),
    ("linesize", ctypes.c_int * 8),
    ("extended_data", ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))),
    ("width", ctypes.c_int),
    ("height", ctypes.c_int),
    ("nb_samples", ctypes.c_int),
    ("format", ctypes.c_int),
  ]


def _shared_library(name: str) -> str:
  if os.name == "posix" and os.uname().sysname == "Darwin":
    patterns = (f"lib{name}.*.dylib", f"lib{name}.dylib")
  else:
    patterns = (f"lib{name}.so.*", f"lib{name}.so")

  for pattern in patterns:
    # The unversioned files in comma-deps can be linker scripts. Prefer a
    # versioned binary and reject tiny linker scripts when falling back.
    candidates = sorted(glob.glob(os.path.join(ffmpeg.LIB_DIR, pattern)), reverse=True)
    for candidate in candidates:
      if os.path.getsize(candidate) > 4096:
        return candidate
  raise ImportError(f"comma-deps-ffmpeg does not provide shared lib{name}")


def _load_libraries():
  avutil = ctypes.CDLL(_shared_library("avutil"), mode=ctypes.RTLD_GLOBAL)
  avcodec = ctypes.CDLL(_shared_library("avcodec"), mode=ctypes.RTLD_GLOBAL)
  swscale = ctypes.CDLL(_shared_library("swscale"), mode=ctypes.RTLD_GLOBAL)

  avutil.av_log_set_level.argtypes = [ctypes.c_int]
  avutil.av_log_set_level.restype = None
  avutil.av_opt_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
  avutil.av_opt_set.restype = ctypes.c_int
  avutil.av_strerror.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
  avutil.av_strerror.restype = ctypes.c_int

  avcodec.avcodec_find_decoder_by_name.argtypes = [ctypes.c_char_p]
  avcodec.avcodec_find_decoder_by_name.restype = ctypes.c_void_p
  avcodec.avcodec_alloc_context3.argtypes = [ctypes.c_void_p]
  avcodec.avcodec_alloc_context3.restype = ctypes.c_void_p
  avcodec.avcodec_open2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  avcodec.avcodec_open2.restype = ctypes.c_int
  avcodec.avcodec_free_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  avcodec.avcodec_free_context.restype = None
  avcodec.avcodec_flush_buffers.argtypes = [ctypes.c_void_p]
  avcodec.avcodec_flush_buffers.restype = None
  avcodec.avcodec_send_packet.argtypes = [ctypes.c_void_p, ctypes.POINTER(AVPacket)]
  avcodec.avcodec_send_packet.restype = ctypes.c_int
  avcodec.avcodec_receive_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(AVFrame)]
  avcodec.avcodec_receive_frame.restype = ctypes.c_int
  avcodec.av_packet_alloc.argtypes = []
  avcodec.av_packet_alloc.restype = ctypes.POINTER(AVPacket)
  avcodec.av_packet_free.argtypes = [ctypes.POINTER(ctypes.POINTER(AVPacket))]
  avcodec.av_packet_free.restype = None
  avcodec.av_frame_alloc.argtypes = []
  avcodec.av_frame_alloc.restype = ctypes.POINTER(AVFrame)
  avcodec.av_frame_free.argtypes = [ctypes.POINTER(ctypes.POINTER(AVFrame))]
  avcodec.av_frame_free.restype = None
  avcodec.av_frame_unref.argtypes = [ctypes.POINTER(AVFrame)]
  avcodec.av_frame_unref.restype = None

  data_array = ctypes.POINTER(ctypes.c_uint8) * 4
  linesize_array = ctypes.c_int * 4
  swscale.sws_getCachedContext.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
  ]
  swscale.sws_getCachedContext.restype = ctypes.c_void_p
  swscale.sws_scale.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_int),
  ]
  swscale.sws_scale.restype = ctypes.c_int
  swscale.sws_freeContext.argtypes = [ctypes.c_void_p]
  swscale.sws_freeContext.restype = None

  avutil.av_log_set_level(AV_LOG_QUIET)
  return avutil, avcodec, swscale, data_array, linesize_array


_avutil, _avcodec, _swscale, _DataArray, _LinesizeArray = _load_libraries()


def _error_string(code: int) -> str:
  buf = ctypes.create_string_buffer(256)
  if _avutil.av_strerror(code, buf, len(buf)) == 0:
    return buf.value.decode(errors="replace")
  return f"FFmpeg error {code}"


def _check(code: int, operation: str) -> None:
  if code < 0:
    raise FFmpegError(f"{operation}: {_error_string(code)}")


class Decoder:
  """Minimal, synchronous HEVC decoder returning reusable NV12 frames.

  A yielded frame is valid until the generator advances. Consumers must use or
  copy it before requesting the next frame. This lets the camera stream hand the
  buffer directly to VisionIPC without allocating per frame.
  """

  def __init__(self, codec_name: str = "hevc"):
    self._context = ctypes.c_void_p()
    self._packet = None
    self._frame = None
    self._sws_context = ctypes.c_void_p()
    self.closed = False

    codec = _avcodec.avcodec_find_decoder_by_name(codec_name.encode())
    if not codec:
      raise FFmpegError(f"decoder not found: {codec_name}")

    self._context = ctypes.c_void_p(_avcodec.avcodec_alloc_context3(codec))
    if not self._context:
      raise MemoryError("avcodec_alloc_context3 failed")

    self._packet = _avcodec.av_packet_alloc()
    self._frame = _avcodec.av_frame_alloc()
    self._packet_buffer = bytearray()
    self._output = np.empty(0, dtype=np.uint8)
    self._dst_data = _DataArray()
    self._dst_linesize = _LinesizeArray()
    self.width = 0
    self.height = 0
    if not self._packet or not self._frame:
      self.close()
      raise MemoryError("failed to allocate FFmpeg packet or frame")

    try:
      # Frame threading holds decoded frames to populate worker pipelines.
      # Slice threads can reduce decode time without adding that frame queue;
      # four was the latency minimum on the replay camera workload.
      _check(_avutil.av_opt_set(self._context, b"threads", b"4", 0), "set decoder threads")
      _check(_avutil.av_opt_set(self._context, b"thread_type", b"slice", 0), "set decoder thread type")
      _check(_avutil.av_opt_set(self._context, b"flags", b"+low_delay", 0), "set low-delay mode")
      _check(_avcodec.avcodec_open2(self._context, codec, None), "open decoder")
    except Exception:
      self.close()
      raise

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def _ensure_open(self) -> None:
    if self.closed:
      raise RuntimeError("decoder is closed")

  def _prepare_packet(self, data) -> None:
    view = memoryview(data)
    if view.ndim != 1 or not view.contiguous:
      view = memoryview(bytes(view))
    size = view.nbytes
    required = size + AV_INPUT_BUFFER_PADDING_SIZE
    if len(self._packet_buffer) < required:
      self._packet_buffer = bytearray(required)

    packet_address = ctypes.addressof(ctypes.c_uint8.from_buffer(self._packet_buffer))
    if size:
      self._packet_buffer[:size] = view
    ctypes.memset(packet_address + size, 0, AV_INPUT_BUFFER_PADDING_SIZE)
    self._packet.contents.data = ctypes.cast(packet_address, ctypes.POINTER(ctypes.c_uint8))
    self._packet.contents.size = size

  def _prepare_output(self, frame: AVFrame) -> None:
    width, height = frame.width, frame.height
    if width <= 0 or height <= 0 or width % 2 or height % 2:
      raise FFmpegError(f"unsupported frame dimensions: {width}x{height}")

    if width != self.width or height != self.height:
      self.width, self.height = width, height
      self._output = np.empty(width * height * 3 // 2, dtype=np.uint8)
      output_address = self._output.ctypes.data
      self._dst_data = _DataArray(
        ctypes.cast(output_address, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(output_address + width * height, ctypes.POINTER(ctypes.c_uint8)),
        None,
        None,
      )
      self._dst_linesize = _LinesizeArray(width, width, 0, 0)

    sws_context = _swscale.sws_getCachedContext(
      self._sws_context, width, height, frame.format,
      width, height, AV_PIX_FMT_NV12, SWS_FAST_BILINEAR,
      None, None, None,
    )
    if not sws_context:
      raise FFmpegError("sws_getCachedContext failed")
    self._sws_context = ctypes.c_void_p(sws_context)

  def _receive(self) -> Generator[np.ndarray, None, None]:
    while True:
      result = _avcodec.avcodec_receive_frame(self._context, self._frame)
      if result in (-errno.EAGAIN, AVERROR_EOF):
        return
      _check(result, "receive decoded frame")

      frame = self._frame.contents
      self._prepare_output(frame)
      rows = _swscale.sws_scale(
        self._sws_context, frame.data, frame.linesize, 0, frame.height,
        self._dst_data, self._dst_linesize,
      )
      if rows != frame.height:
        _avcodec.av_frame_unref(self._frame)
        raise FFmpegError(f"convert decoded frame: produced {rows} of {frame.height} rows")
      try:
        yield self._output
      finally:
        _avcodec.av_frame_unref(self._frame)

  def decode(self, data) -> Generator[np.ndarray, None, None]:
    self._ensure_open()
    if memoryview(data).nbytes == 0:
      return

    self._prepare_packet(data)
    result = _avcodec.avcodec_send_packet(self._context, self._packet)
    # The packet buffer is ours, not FFmpeg's. Clear the borrowed pointer so
    # packet teardown can never attempt to release it.
    self._packet.contents.data = None
    self._packet.contents.size = 0
    _check(result, "send packet to decoder")
    yield from self._receive()

  def flush(self) -> Generator[np.ndarray, None, None]:
    """Drain delayed frames at end of stream."""
    self._ensure_open()
    _check(_avcodec.avcodec_send_packet(self._context, None), "flush decoder")
    yield from self._receive()

  def reset(self) -> None:
    """Discard decoder state after a stream discontinuity."""
    self._ensure_open()
    _avcodec.avcodec_flush_buffers(self._context)

  def close(self) -> None:
    if self.closed:
      return
    self.closed = True
    if self._sws_context:
      _swscale.sws_freeContext(self._sws_context)
      self._sws_context = ctypes.c_void_p()
    if self._frame:
      _avcodec.av_frame_free(ctypes.byref(self._frame))
    if self._packet:
      _avcodec.av_packet_free(ctypes.byref(self._packet))
    if self._context:
      _avcodec.avcodec_free_context(ctypes.byref(self._context))

  def __del__(self):
    self.close()
