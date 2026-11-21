import ctypes
import errno
import os

import ffmpeg
import numpy as np


AV_INPUT_BUFFER_PADDING_SIZE = 64
AV_LOG_QUIET = -8
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
  # Public prefix of AVFrame through format. Stable within a libavutil major
  _fields_ = [
    ("data", ctypes.POINTER(ctypes.c_uint8) * 8),
    ("linesize", ctypes.c_int * 8),
    ("extended_data", ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))),
    ("width", ctypes.c_int),
    ("height", ctypes.c_int),
    ("nb_samples", ctypes.c_int),
    ("format", ctypes.c_int),
  ]


def _bind(fn, restype, *argtypes):
  fn.restype = restype
  fn.argtypes = list(argtypes)
  return fn


def _load_libraries():
  avutil = ctypes.CDLL(os.path.join(ffmpeg.LIB_DIR, "libavutil.so.59"), mode=ctypes.RTLD_GLOBAL)
  avcodec = ctypes.CDLL(os.path.join(ffmpeg.LIB_DIR, "libavcodec.so.61"), mode=ctypes.RTLD_GLOBAL)
  swscale = ctypes.CDLL(os.path.join(ffmpeg.LIB_DIR, "libswscale.so.8"), mode=ctypes.RTLD_GLOBAL)

  c_int, c_char_p, c_void_p, c_size_t = ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t
  c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
  c_void_p_p = ctypes.POINTER(c_void_p)

  _bind(avutil.av_log_set_level, None, c_int)
  _bind(avutil.av_opt_set, c_int, c_void_p, c_char_p, c_char_p, c_int)
  _bind(avutil.av_strerror, c_int, c_int, c_char_p, c_size_t)
  _bind(avutil.av_get_pix_fmt, c_int, c_char_p)

  _bind(avcodec.avcodec_find_decoder_by_name, c_void_p, c_char_p)
  _bind(avcodec.avcodec_alloc_context3, c_void_p, c_void_p)
  _bind(avcodec.avcodec_open2, c_int, c_void_p, c_void_p, c_void_p)
  _bind(avcodec.avcodec_free_context, None, c_void_p_p)
  _bind(avcodec.avcodec_flush_buffers, None, c_void_p)
  _bind(avcodec.avcodec_send_packet, c_int, c_void_p, ctypes.POINTER(AVPacket))
  _bind(avcodec.avcodec_receive_frame, c_int, c_void_p, ctypes.POINTER(AVFrame))
  _bind(avcodec.av_packet_alloc, ctypes.POINTER(AVPacket))
  _bind(avcodec.av_packet_free, None, ctypes.POINTER(ctypes.POINTER(AVPacket)))
  _bind(avcodec.av_frame_alloc, ctypes.POINTER(AVFrame))
  _bind(avcodec.av_frame_free, None, ctypes.POINTER(ctypes.POINTER(AVFrame)))
  _bind(avcodec.av_frame_unref, None, ctypes.POINTER(AVFrame))

  _bind(swscale.sws_getCachedContext, c_void_p,
        c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p)
  _bind(swscale.sws_scale, c_int,
        c_void_p, ctypes.POINTER(c_uint8_p), ctypes.POINTER(c_int),
        c_int, c_int, ctypes.POINTER(c_uint8_p), ctypes.POINTER(c_int))
  _bind(swscale.sws_freeContext, None, c_void_p)

  avutil.av_log_set_level(AV_LOG_QUIET)
  return avutil, avcodec, swscale
_avutil, _avcodec, _swscale = _load_libraries()

_DataArray = ctypes.POINTER(ctypes.c_uint8) * 4
_LinesizeArray = ctypes.c_int * 4

AV_PIX_FMT_NV12 = _avutil.av_get_pix_fmt(b"nv12")
assert AV_PIX_FMT_NV12 >= 0


def _error_string(code: int) -> str:
  buf = ctypes.create_string_buffer(256)
  if _avutil.av_strerror(code, buf, len(buf)) == 0:
    return buf.value.decode(errors="replace")
  return f"FFmpeg error {code}"


def _check(code: int, operation: str) -> None:
  if code < 0:
    raise FFmpegError(f"{operation}: {_error_string(code)}")


class Decoder:
  def __init__(self, codec_name: str = "hevc"):
    self.closed = True
    self._sws_context = ctypes.c_void_p()
    self._packet_buffer = bytearray()
    self._packet_address = 0
    self._packet_data = None
    self._output = np.empty(0, dtype=np.uint8)
    self._dst_data = _DataArray()
    self._dst_linesize = _LinesizeArray()
    self.width = 0
    self.height = 0
    self._src_format = -1

    codec = _avcodec.avcodec_find_decoder_by_name(codec_name.encode())
    if not codec:
      raise FFmpegError(f"decoder not found: {codec_name}")

    self._context = ctypes.c_void_p(_avcodec.avcodec_alloc_context3(codec))
    if not self._context:
      raise MemoryError("avcodec_alloc_context3 failed")

    self._packet = _avcodec.av_packet_alloc()
    if not self._packet:
      _avcodec.avcodec_free_context(ctypes.byref(self._context))
      raise MemoryError("av_packet_alloc failed")

    self._frame = _avcodec.av_frame_alloc()
    if not self._frame:
      _avcodec.av_packet_free(ctypes.byref(self._packet))
      _avcodec.avcodec_free_context(ctypes.byref(self._context))
      raise MemoryError("av_frame_alloc failed")

    try:
      # Frame threading holds decoded frames to populate worker pipelines.
      # Slice threads can reduce decode time without adding that frame queue;
      # four was the latency minimum on the replay camera workload.
      _check(_avutil.av_opt_set(self._context, b"threads", b"4", 0), "set decoder threads")
      _check(_avutil.av_opt_set(self._context, b"thread_type", b"slice", 0), "set decoder thread type")
      _check(_avutil.av_opt_set(self._context, b"flags", b"+low_delay", 0), "set low-delay mode")
      _check(_avcodec.avcodec_open2(self._context, codec, None), "open decoder")
    except Exception:
      _avcodec.av_frame_free(ctypes.byref(self._frame))
      _avcodec.av_packet_free(ctypes.byref(self._packet))
      _avcodec.avcodec_free_context(ctypes.byref(self._context))
      raise

    self.closed = False

  def __enter__(self):
    self._ensure_open()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def _ensure_open(self) -> None:
    if self.closed:
      raise RuntimeError("decoder is closed")

  def _prepare_packet(self, data) -> None:
    size = len(data)
    required = size + AV_INPUT_BUFFER_PADDING_SIZE
    if len(self._packet_buffer) < required:
      # Grow-only; the address stays valid until the next reallocation.
      self._packet_buffer = bytearray(required)
      self._packet_address = ctypes.addressof(ctypes.c_uint8.from_buffer(self._packet_buffer))
      self._packet_data = ctypes.cast(self._packet_address, ctypes.POINTER(ctypes.c_uint8))

    self._packet_buffer[:size] = data
    ctypes.memset(self._packet_address + size, 0, AV_INPUT_BUFFER_PADDING_SIZE)
    self._packet.contents.data = self._packet_data
    self._packet.contents.size = size

  def _prepare_output(self, frame: AVFrame) -> None:
    width, height, src_format = frame.width, frame.height, frame.format
    if width <= 0 or height <= 0 or width % 2 or height % 2:
      raise FFmpegError(f"unsupported frame dimensions: {width}x{height}")
    if (width, height, src_format) == (self.width, self.height, self._src_format):
      return

    sws_context = _swscale.sws_getCachedContext(
      self._sws_context, width, height, src_format,
      width, height, AV_PIX_FMT_NV12, SWS_FAST_BILINEAR,
      None, None, None,
    )
    if not sws_context:
      raise FFmpegError("sws_getCachedContext failed")
    self._sws_context = ctypes.c_void_p(sws_context)

    self.width, self.height = width, height
    self._src_format = src_format
    self._output = np.empty(width * height * 3 // 2, dtype=np.uint8)
    output_address = self._output.ctypes.data
    self._dst_data = _DataArray(
      ctypes.cast(output_address, ctypes.POINTER(ctypes.c_uint8)),
      ctypes.cast(output_address + width * height, ctypes.POINTER(ctypes.c_uint8)),
      None,
      None,
    )
    self._dst_linesize = _LinesizeArray(width, width, 0, 0)

  def _receive(self) -> np.ndarray | None:
    """Return one NV12 frame, or None if the decoder needs more input.

    The returned buffer is reused on the next successful decode; callers must
    use or copy it before calling decode again.
    """
    result = _avcodec.avcodec_receive_frame(self._context, self._frame)
    if result == -errno.EAGAIN:
      return None
    _check(result, "receive decoded frame")

    try:
      frame = self._frame.contents
      self._prepare_output(frame)
      rows = _swscale.sws_scale(
        self._sws_context, frame.data, frame.linesize, 0, frame.height,
        self._dst_data, self._dst_linesize,
      )
      if rows != frame.height:
        raise FFmpegError(f"convert decoded frame: produced {rows} of {frame.height} rows")
      return self._output
    finally:
      _avcodec.av_frame_unref(self._frame)

  def decode(self, data) -> np.ndarray | None:
    self._ensure_open()
    if len(data) == 0:
      return None

    self._prepare_packet(data)
    result = _avcodec.avcodec_send_packet(self._context, self._packet)
    # The packet buffer is ours, not FFmpeg's. Clear the borrowed pointer so
    # packet teardown can never attempt to release it.
    self._packet.contents.data = None
    self._packet.contents.size = 0
    _check(result, "send packet to decoder")
    return self._receive()

  def reset(self) -> None:
    """Discard decoder state after a stream discontinuity."""
    self._ensure_open()
    _avcodec.avcodec_flush_buffers(self._context)

  def close(self) -> None:
    if self.closed:
      return
    self.closed = True
    _swscale.sws_freeContext(self._sws_context)
    _avcodec.av_frame_free(ctypes.byref(self._frame))
    _avcodec.av_packet_free(ctypes.byref(self._packet))
    _avcodec.avcodec_free_context(ctypes.byref(self._context))

  def __del__(self):
    self.close()
