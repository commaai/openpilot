import ctypes
import ctypes.util
import os
import platform
import subprocess
import json
import logging
from collections.abc import Iterator
from collections import OrderedDict

import numpy as np
from openpilot.tools.lib.filereader import FileReader, resolve_name
from openpilot.tools.lib.exceptions import DataUnreadableError
from openpilot.tools.lib.vidindex import hevc_index

logger = logging.getLogger("tools")

HEVC_SLICE_B = 0
HEVC_SLICE_P = 1
HEVC_SLICE_I = 2

class LRUCache:
  def __init__(self, capacity: int):
    self._cache: OrderedDict = OrderedDict()
    self.capacity = capacity

  def __getitem__(self, key):
    self._cache.move_to_end(key)
    return self._cache[key]

  def __setitem__(self, key, value):
    self._cache[key] = value
    if len(self._cache) > self.capacity:
        self._cache.popitem(last=False)

  def __contains__(self, key):
    return key in self._cache

def assert_hvec(fn: str) -> None:
  with FileReader(fn) as f:
    header = f.read(4)
  if len(header) == 0:
    raise DataUnreadableError(f"{fn} is empty")
  elif header == b"\x00\x00\x00\x01":
    if 'hevc' not in fn:
      raise NotImplementedError(fn)

def decompress_video_data(rawdat, w, h, pix_fmt="rgb24", vid_fmt='hevc', hwaccel="auto", loglevel="info") -> np.ndarray:
  threads = os.getenv("FFMPEG_THREADS", "0")
  args = ["ffmpeg", "-v", loglevel,
          "-threads", threads,
          "-hwaccel", hwaccel,
          "-c:v", "hevc",
          "-vsync", "0",
          "-f", vid_fmt,
          "-flags2", "showall",
          "-i", "pipe:0",
          "-f", "rawvideo",
          "-pix_fmt", pix_fmt,
          "pipe:1"]
  dat = subprocess.check_output(args, input=rawdat)

  ret: np.ndarray
  if pix_fmt == "rgb24":
    ret = np.frombuffer(dat, dtype=np.uint8).reshape(-1, h, w, 3)
  elif pix_fmt in ["nv12", "yuv420p"]:
    ret = np.frombuffer(dat, dtype=np.uint8).reshape(-1, (h*w*3//2))
  else:
    raise NotImplementedError(f"Unsupported pixel format: {pix_fmt}")
  return ret

_AV_CODEC_ID_HEVC = 173
_AV_CODEC_FLAG_LOW_DELAY = 1 << 19
_AV_CODEC_FLAG2_SHOW_ALL = 1 << 22
_AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX = 0x01
_FF_THREAD_SLICE = 2

_AV_LOG_QUIET = -8
_AVERROR_EAGAIN = -11
_AVERROR_EOF = -541478725

_AV_PIX_FMT_NONE = -1
_AV_PIX_FMT_YUV420P = 0
_AV_PIX_FMT_YUVJ420P = 12
_AV_PIX_FMT_NV12 = 23

_AV_HWDEVICE_TYPE_CUDA = 2
_AV_HWDEVICE_TYPE_VIDEOTOOLBOX = 6
_HW_DEVICE_TYPE = _AV_HWDEVICE_TYPE_VIDEOTOOLBOX if platform.system() == "Darwin" else _AV_HWDEVICE_TYPE_CUDA

_AVCODEC_VERSION_MAJOR = 60
_AVUTIL_VERSION_MAJOR = 58

# Minimal FFmpeg 6 struct definitions for fields touched by the streaming
# decoder. These are guarded by avcodec_version()/avutil_version() at load time.
class _AVCodecContext(ctypes.Structure):
  _fields_ = [
    ("_pad0", ctypes.c_uint8 * 48),
    ("opaque", ctypes.c_void_p),
    ("_pad1", ctypes.c_uint8 * 20),
    ("flags", ctypes.c_int),
    ("flags2", ctypes.c_int),
    ("_pad2", ctypes.c_uint8 * 32),
    ("width", ctypes.c_int),
    ("height", ctypes.c_int),
    ("_pad3", ctypes.c_uint8 * 28),
    ("get_format", ctypes.c_void_p),
    ("_pad4", ctypes.c_uint8 * 476),
    ("thread_count", ctypes.c_int),
    ("thread_type", ctypes.c_int),
    ("_pad5", ctypes.c_uint8 * 220),
    ("hw_device_ctx", ctypes.c_void_p),
  ]

class _AVCodecHWConfig(ctypes.Structure):
  _fields_ = [
    ("pix_fmt", ctypes.c_int),
    ("methods", ctypes.c_int),
    ("device_type", ctypes.c_int),
  ]

class _AVFrame(ctypes.Structure):
  _fields_ = [
    ("data", ctypes.c_void_p * 8),
    ("linesize", ctypes.c_int * 8),
    ("extended_data", ctypes.POINTER(ctypes.c_void_p)),
    ("width", ctypes.c_int),
    ("height", ctypes.c_int),
    ("nb_samples", ctypes.c_int),
    ("format", ctypes.c_int),
  ]

class _AVPacket(ctypes.Structure):
  _fields_ = [
    ("buf", ctypes.c_void_p),
    ("pts", ctypes.c_int64),
    ("dts", ctypes.c_int64),
    ("data", ctypes.POINTER(ctypes.c_uint8)),
    ("size", ctypes.c_int),
  ]

_GET_FORMAT_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_AVCodecContext), ctypes.POINTER(ctypes.c_int))

class _Libavcodec:
  def __init__(self):
    avcodec_path = ctypes.util.find_library("avcodec")
    avutil_path = ctypes.util.find_library("avutil")
    if avcodec_path is None or avutil_path is None:
      raise RuntimeError("libavcodec/libavutil shared libraries were not found")

    self.avcodec = ctypes.CDLL(avcodec_path)
    self.avutil = ctypes.CDLL(avutil_path)

    self._bind_versions()
    self._check_abi()
    self._bind_avutil()
    self._bind_avcodec()
    self.avutil.av_log_set_level(_AV_LOG_QUIET)

  def _bind_versions(self) -> None:
    self.avcodec.avcodec_version.restype = ctypes.c_uint
    self.avutil.avutil_version.restype = ctypes.c_uint

  def _check_abi(self) -> None:
    avcodec_major = self.avcodec.avcodec_version() >> 16
    avutil_major = self.avutil.avutil_version() >> 16
    if avcodec_major != _AVCODEC_VERSION_MAJOR or avutil_major != _AVUTIL_VERSION_MAJOR:
      raise RuntimeError("ctypes libavcodec decoder expects FFmpeg 6 shared libraries")

  def _bind_avutil(self) -> None:
    self.avutil.av_log_set_level.argtypes = [ctypes.c_int]
    self.avutil.av_log_set_level.restype = None
    self.avutil.av_strerror.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    self.avutil.av_strerror.restype = ctypes.c_int
    self.avutil.av_buffer_ref.argtypes = [ctypes.c_void_p]
    self.avutil.av_buffer_ref.restype = ctypes.c_void_p
    self.avutil.av_buffer_unref.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    self.avutil.av_buffer_unref.restype = None
    self.avutil.av_frame_alloc.argtypes = []
    self.avutil.av_frame_alloc.restype = ctypes.POINTER(_AVFrame)
    self.avutil.av_frame_free.argtypes = [ctypes.POINTER(ctypes.POINTER(_AVFrame))]
    self.avutil.av_frame_free.restype = None
    self.avutil.av_frame_unref.argtypes = [ctypes.POINTER(_AVFrame)]
    self.avutil.av_frame_unref.restype = None
    self.avutil.av_hwdevice_ctx_create.argtypes = [
      ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int
    ]
    self.avutil.av_hwdevice_ctx_create.restype = ctypes.c_int
    self.avutil.av_hwframe_transfer_data.argtypes = [ctypes.POINTER(_AVFrame), ctypes.POINTER(_AVFrame), ctypes.c_int]
    self.avutil.av_hwframe_transfer_data.restype = ctypes.c_int

  def _bind_avcodec(self) -> None:
    self.avcodec.avcodec_find_decoder.argtypes = [ctypes.c_int]
    self.avcodec.avcodec_find_decoder.restype = ctypes.c_void_p
    self.avcodec.avcodec_alloc_context3.argtypes = [ctypes.c_void_p]
    self.avcodec.avcodec_alloc_context3.restype = ctypes.POINTER(_AVCodecContext)
    self.avcodec.avcodec_free_context.argtypes = [ctypes.POINTER(ctypes.POINTER(_AVCodecContext))]
    self.avcodec.avcodec_free_context.restype = None
    self.avcodec.avcodec_open2.argtypes = [ctypes.POINTER(_AVCodecContext), ctypes.c_void_p, ctypes.c_void_p]
    self.avcodec.avcodec_open2.restype = ctypes.c_int
    self.avcodec.avcodec_get_hw_config.argtypes = [ctypes.c_void_p, ctypes.c_int]
    self.avcodec.avcodec_get_hw_config.restype = ctypes.POINTER(_AVCodecHWConfig)
    self.avcodec.avcodec_send_packet.argtypes = [ctypes.POINTER(_AVCodecContext), ctypes.POINTER(_AVPacket)]
    self.avcodec.avcodec_send_packet.restype = ctypes.c_int
    self.avcodec.avcodec_receive_frame.argtypes = [ctypes.POINTER(_AVCodecContext), ctypes.POINTER(_AVFrame)]
    self.avcodec.avcodec_receive_frame.restype = ctypes.c_int
    self.avcodec.av_packet_alloc.argtypes = []
    self.avcodec.av_packet_alloc.restype = ctypes.POINTER(_AVPacket)
    self.avcodec.av_packet_free.argtypes = [ctypes.POINTER(ctypes.POINTER(_AVPacket))]
    self.avcodec.av_packet_free.restype = None
    self.avcodec.av_new_packet.argtypes = [ctypes.POINTER(_AVPacket), ctypes.c_int]
    self.avcodec.av_new_packet.restype = ctypes.c_int

  def error(self, err: int) -> str:
    errbuf = ctypes.create_string_buffer(128)
    self.avutil.av_strerror(err, errbuf, len(errbuf))
    return errbuf.value.decode("utf-8", "replace")

_libavcodec: _Libavcodec | None = None

def _libav() -> _Libavcodec:
  global _libavcodec
  if _libavcodec is None:
    _libavcodec = _Libavcodec()
  return _libavcodec

def _plane_array(data: int, linesize: int, rows: int, cols: int) -> np.ndarray:
  if not data:
    raise RuntimeError("decoded frame has an empty data plane")
  if linesize < cols:
    raise RuntimeError("decoded frame linesize is smaller than expected")
  plane = (ctypes.c_uint8 * (linesize * rows)).from_address(data)
  return np.ctypeslib.as_array(plane).reshape(rows, linesize)[:, :cols]

def _copy_frame_to_nv12(frame: _AVFrame, width: int, height: int) -> np.ndarray:
  if frame.width < width or frame.height < height:
    raise RuntimeError("decoded frame is smaller than expected")

  y_size = width * height
  out = np.empty(y_size * 3 // 2, dtype=np.uint8)
  out_y = out[:y_size].reshape(height, width)
  out_uv = out[y_size:].reshape(height // 2, width)

  if frame.format == _AV_PIX_FMT_NV12:
    out_y[:] = _plane_array(frame.data[0], frame.linesize[0], height, width)
    out_uv[:] = _plane_array(frame.data[1], frame.linesize[1], height // 2, width)
  elif frame.format in (_AV_PIX_FMT_YUV420P, _AV_PIX_FMT_YUVJ420P):
    out_y[:] = _plane_array(frame.data[0], frame.linesize[0], height, width)
    out_uv[:, 0::2] = _plane_array(frame.data[1], frame.linesize[1], height // 2, width // 2)
    out_uv[:, 1::2] = _plane_array(frame.data[2], frame.linesize[2], height // 2, width // 2)
  else:
    raise RuntimeError(f"unsupported decoded pixel format: {frame.format}")
  return out

class LibavcodecStreamDecoder:
  def __init__(self, w: int, h: int, pix_fmt: str = "nv12", vid_fmt: str = "hevc", hwaccel: str = "auto"):
    if pix_fmt != "nv12":
      raise NotImplementedError(f"Unsupported pixel format: {pix_fmt}")
    if vid_fmt != "hevc":
      raise NotImplementedError(f"Unsupported video format: {vid_fmt}")

    self.w = w
    self.h = h
    self.lib = _libav()
    self.decoder_ctx = ctypes.POINTER(_AVCodecContext)()
    self.frame = ctypes.POINTER(_AVFrame)()
    self.sw_frame = ctypes.POINTER(_AVFrame)()
    self.hw_device_ctx = ctypes.c_void_p()
    self.hw_pix_fmt = _AV_PIX_FMT_NONE
    self._frames: list[np.ndarray] = []
    self._get_format_cb = None

    decoder = self.lib.avcodec.avcodec_find_decoder(_AV_CODEC_ID_HEVC)
    if not decoder:
      raise RuntimeError("HEVC decoder not found")

    self.decoder_ctx = self.lib.avcodec.avcodec_alloc_context3(decoder)
    self.frame = self.lib.avutil.av_frame_alloc()
    self.sw_frame = self.lib.avutil.av_frame_alloc()
    if not self.decoder_ctx or not self.frame or not self.sw_frame:
      self.close()
      raise RuntimeError("failed to allocate decoder state")

    ctx = self.decoder_ctx.contents
    ctx.width = w
    ctx.height = h
    ctx.thread_count = max(1, int(os.getenv("FFMPEG_STREAM_THREADS", "1")))
    ctx.thread_type = _FF_THREAD_SLICE
    ctx.flags |= _AV_CODEC_FLAG_LOW_DELAY
    ctx.flags2 |= _AV_CODEC_FLAG2_SHOW_ALL

    if hwaccel != "none":
      self._init_hardware_decoder(decoder)

    ret = self.lib.avcodec.avcodec_open2(self.decoder_ctx, decoder, None)
    if ret < 0:
      self.close()
      raise RuntimeError(f"failed to open HEVC decoder: {self.lib.error(ret)}")

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self) -> None:
    if self.hw_device_ctx:
      self.lib.avutil.av_buffer_unref(ctypes.byref(self.hw_device_ctx))
    if self.decoder_ctx:
      self.lib.avcodec.avcodec_free_context(ctypes.byref(self.decoder_ctx))
    if self.frame:
      self.lib.avutil.av_frame_free(ctypes.byref(self.frame))
    if self.sw_frame:
      self.lib.avutil.av_frame_free(ctypes.byref(self.sw_frame))
    self.hw_pix_fmt = _AV_PIX_FMT_NONE

  def _get_format(self, _ctx, pix_fmts) -> int:
    i = 0
    while pix_fmts[i] != _AV_PIX_FMT_NONE:
      if pix_fmts[i] == self.hw_pix_fmt:
        return self.hw_pix_fmt
      i += 1
    self.hw_pix_fmt = _AV_PIX_FMT_NONE
    return _AV_PIX_FMT_YUV420P

  def _init_hardware_decoder(self, decoder: int) -> None:
    i = 0
    while True:
      config = self.lib.avcodec.avcodec_get_hw_config(decoder, i)
      if not config:
        return
      if (config.contents.methods & _AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX and
          config.contents.device_type == _HW_DEVICE_TYPE):
        self.hw_pix_fmt = config.contents.pix_fmt
        break
      i += 1

    ret = self.lib.avutil.av_hwdevice_ctx_create(ctypes.byref(self.hw_device_ctx), _HW_DEVICE_TYPE, None, None, 0)
    if ret < 0:
      self.hw_pix_fmt = _AV_PIX_FMT_NONE
      return

    self.decoder_ctx.contents.hw_device_ctx = self.lib.avutil.av_buffer_ref(self.hw_device_ctx)
    if not self.decoder_ctx.contents.hw_device_ctx:
      self.hw_pix_fmt = _AV_PIX_FMT_NONE
      self.lib.avutil.av_buffer_unref(ctypes.byref(self.hw_device_ctx))
      return

    self._get_format_cb = _GET_FORMAT_CB(self._get_format)
    self.decoder_ctx.contents.get_format = ctypes.cast(self._get_format_cb, ctypes.c_void_p).value

  def _receive_frames(self) -> None:
    while True:
      ret = self.lib.avcodec.avcodec_receive_frame(self.decoder_ctx, self.frame)
      if ret in (_AVERROR_EAGAIN, _AVERROR_EOF):
        return
      if ret < 0:
        raise RuntimeError(f"failed to receive decoded frame: {self.lib.error(ret)}")

      out_frame = self.frame
      if self.frame.contents.format == self.hw_pix_fmt:
        self.lib.avutil.av_frame_unref(self.sw_frame)
        ret = self.lib.avutil.av_hwframe_transfer_data(self.sw_frame, self.frame, 0)
        if ret < 0:
          self.lib.avutil.av_frame_unref(self.frame)
          raise RuntimeError(f"failed to transfer hardware frame: {self.lib.error(ret)}")
        out_frame = self.sw_frame

      self._frames.append(_copy_frame_to_nv12(out_frame.contents, self.w, self.h))
      self.lib.avutil.av_frame_unref(self.frame)
      self.lib.avutil.av_frame_unref(self.sw_frame)

  def decode(self, packet: bytes | bytearray | memoryview) -> np.ndarray | None:
    packet_bytes = packet if isinstance(packet, bytes) else bytes(packet)
    if len(packet_bytes) == 0:
      return None

    pkt = self.lib.avcodec.av_packet_alloc()
    if not pkt:
      raise RuntimeError("failed to allocate packet")

    try:
      ret = self.lib.avcodec.av_new_packet(pkt, len(packet_bytes))
      if ret < 0:
        raise RuntimeError(f"failed to allocate packet data: {self.lib.error(ret)}")
      ctypes.memmove(pkt.contents.data, packet_bytes, len(packet_bytes))

      ret = self.lib.avcodec.avcodec_send_packet(self.decoder_ctx, pkt)
      if ret == _AVERROR_EAGAIN:
        self._receive_frames()
        ret = self.lib.avcodec.avcodec_send_packet(self.decoder_ctx, pkt)
      if ret < 0:
        raise RuntimeError(f"failed to send packet to decoder: {self.lib.error(ret)}")
    finally:
      self.lib.avcodec.av_packet_free(ctypes.byref(pkt))

    self._receive_frames()
    if not self._frames:
      return None
    return self._frames.pop(0)

def ffprobe(fn, fmt=None):
  fn = resolve_name(fn)
  cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams"]
  if fmt:
    cmd += ["-f", fmt]
  cmd += ["-i", "pipe:0"]

  try:
    with FileReader(fn) as f:
      ffprobe_output = subprocess.check_output(cmd, input=f.read(4096))
  except subprocess.CalledProcessError as e:
    raise DataUnreadableError(fn) from e
  return json.loads(ffprobe_output)

def get_index_data(fn: str, index_data: dict|None = None):
  if index_data is None:
    index_data = get_video_index(fn)
    if index_data is None:
      raise DataUnreadableError(f"Failed to index {fn!r}")
  stream = index_data["probe"]["streams"][0]
  return index_data["index"], index_data["global_prefix"], stream["width"], stream["height"]

def get_video_index(fn):
  assert_hvec(fn)
  frame_types, dat_len, prefix = hevc_index(fn)
  index = np.array(frame_types + [(0xFFFFFFFF, dat_len)], dtype=np.uint32)
  probe = ffprobe(fn, "hevc")
  return {
    'index': index,
    'global_prefix': prefix,
    'probe': probe
  }

class FfmpegDecoder:
  def __init__(self, fn: str, index_data: dict|None = None,
               pix_fmt: str = "rgb24", hwaccel="auto", loglevel="quiet"):
    self.fn = fn
    self.index, self.prefix, self.w, self.h = get_index_data(fn, index_data)
    self.frame_count = len(self.index) - 1          # sentinel row at the end
    self.iframes = np.where(self.index[:, 0] == HEVC_SLICE_I)[0]
    self.pix_fmt = pix_fmt
    self.loglevel, self.hwaccel = loglevel, hwaccel

  def _gop_bounds(self, frame_idx: int):
    f_b = frame_idx
    while f_b > 0 and self.index[f_b, 0] != HEVC_SLICE_I:
      f_b -= 1
    f_e = frame_idx + 1
    while f_e < self.frame_count and self.index[f_e, 0] != HEVC_SLICE_I:
      f_e += 1
    return f_b, f_e, self.index[f_b, 1], self.index[f_e, 1]

  def _decode_gop(self, raw: bytes) -> Iterator[np.ndarray]:
    yield from decompress_video_data(raw, self.w, self.h, pix_fmt=self.pix_fmt, hwaccel=self.hwaccel, loglevel=self.loglevel)

  def get_gop_start(self, frame_idx: int):
    return self.iframes[np.searchsorted(self.iframes, frame_idx, side="right") - 1]

  def get_iterator(self, start_fidx: int = 0, end_fidx: int|None = None,
                   frame_skip: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    end_fidx = end_fidx or self.frame_count
    fidx = start_fidx
    while fidx < end_fidx:
      f_b, f_e, off_b, off_e = self._gop_bounds(fidx)
      with FileReader(self.fn) as f:
        f.seek(off_b)
        raw = self.prefix + f.read(off_e - off_b)
      # number of frames to discard inside this GOP before the wanted one
      for i, frm in enumerate(decompress_video_data(raw, self.w, self.h, self.pix_fmt, hwaccel=self.hwaccel, loglevel=self.loglevel)):
        fidx = f_b + i
        if fidx >= end_fidx:
          return
        elif fidx >= start_fidx and (fidx - start_fidx) % frame_skip == 0:
          yield fidx, frm
      fidx += 1

def FrameIterator(fn: str, index_data: dict|None=None, pix_fmt: str = "rgb24",
                  start_fidx:int=0, end_fidx=None, frame_skip:int=1, hwaccel="auto", loglevel="quiet") -> Iterator[np.ndarray]:
  dec = FfmpegDecoder(fn, pix_fmt=pix_fmt, index_data=index_data, hwaccel=hwaccel, loglevel=loglevel)
  for _, frame in dec.get_iterator(start_fidx=start_fidx, end_fidx=end_fidx, frame_skip=frame_skip):
    yield frame

class FrameReader:
  def __init__(self, fn: str, index_data: dict|None = None, cache_size: int = 30,
               pix_fmt: str = "rgb24", hwaccel="auto", loglevel="quiet"):
    self.decoder = FfmpegDecoder(fn, index_data=index_data, pix_fmt=pix_fmt, hwaccel=hwaccel, loglevel=loglevel)
    self.iframes = self.decoder.iframes
    self._cache: LRUCache = LRUCache(cache_size)
    self.w, self.h, self.frame_count, = self.decoder.w, self.decoder.h, self.decoder.frame_count
    self.pix_fmt = pix_fmt

    self.it: Iterator[tuple[int, np.ndarray]] | None = None
    self.fidx = -1

  def get(self, fidx:int):
    if fidx in self._cache:  # If frame is cached, return it
      return self._cache[fidx]
    read_start = self.decoder.get_gop_start(fidx)
    if not self.it or fidx < self.fidx or read_start != self.decoder.get_gop_start(self.fidx):  # If the frame is in a different GOP, reset the iterator
      self.it = self.decoder.get_iterator(read_start)
      self.fidx = -1
    while self.fidx < fidx:
      self.fidx, frame = next(self.it)
      self._cache[self.fidx] = frame
    return self._cache[fidx]
