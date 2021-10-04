# pylint: skip-file
import json
import os
import pickle
import struct
import subprocess
import tempfile
import threading
from functools import wraps

import numpy as np
from aenum import Enum
from lru import LRU

import _io
from tools.lib.cache import cache_path_for_file_path
from tools.lib.exceptions import DataUnreadableError
from common.file_helpers import atomic_write_in_dir

try:
  from xx.chffr.lib.filereader import FileReader
except ImportError:
  from tools.lib.filereader import FileReader

HEVC_SLICE_B = 0
HEVC_SLICE_P = 1
HEVC_SLICE_I = 2


class GOPReader:
  def get_gop(self, num):
    # returns (start_frame_num, num_frames, frames_to_skip, gop_data)
    raise NotImplementedError


class DoNothingContextManager:
  def __enter__(self):
    return self

  def __exit__(self, *x):
    pass


class FrameType(Enum):
  raw = 1
  h265_stream = 2


def fingerprint_video(fn):
  with FileReader(fn) as f:
    header = f.read(4)
  if len(header) == 0:
    raise DataUnreadableError(f"{fn} is empty")
  elif header == b"\x00\xc0\x12\x00":
    return FrameType.raw
  elif header == b"\x00\x00\x00\x01":
    if 'hevc' in fn:
      return FrameType.h265_stream
    else:
      raise NotImplementedError(fn)
  else:
    raise NotImplementedError(fn)


def ffprobe(fn, fmt=None):
  cmd = ["ffprobe",
         "-v", "quiet",
         "-print_format", "json",
         "-show_format", "-show_streams"]
  if fmt:
    cmd += ["-f", fmt]
  cmd += [fn]

  try:
    ffprobe_output = subprocess.check_output(cmd)
  except subprocess.CalledProcessError:
    raise DataUnreadableError(fn)

  return json.loads(ffprobe_output)


def vidindex(fn, typ):
  vidindex_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vidindex")
  vidindex = os.path.join(vidindex_dir, "vidindex")

  subprocess.check_call(["make"], cwd=vidindex_dir, stdout=open("/dev/null", "w"))

  with tempfile.NamedTemporaryFile() as prefix_f, \
       tempfile.NamedTemporaryFile() as index_f:
    try:
      subprocess.check_call([vidindex, typ, fn, prefix_f.name, index_f.name])
    except subprocess.CalledProcessError:
      raise DataUnreadableError(f"vidindex failed on file {fn}")
    with open(index_f.name, "rb") as f:
      index = f.read()
    with open(prefix_f.name, "rb") as f:
      prefix = f.read()

  index = np.frombuffer(index, np.uint32).reshape(-1, 2)

  assert index[-1, 0] == 0xFFFFFFFF
  assert index[-1, 1] == os.path.getsize(fn)

  return index, prefix


def cache_fn(func):
  @wraps(func)
  def cache_inner(fn, *args, **kwargs):
    if kwargs.pop('no_cache', None):
      cache_path = None
    else:
      cache_prefix = kwargs.pop('cache_prefix', None)
      cache_path = cache_path_for_file_path(fn, cache_prefix)

    if cache_path and os.path.exists(cache_path):
      with open(cache_path, "rb") as cache_file:
        cache_value = pickle.load(cache_file)
    else:
      cache_value = func(fn, *args, **kwargs)

      if cache_path:
        with atomic_write_in_dir(cache_path, mode="wb", overwrite=True) as cache_file:
          pickle.dump(cache_value, cache_file, -1)

    return cache_value

  return cache_inner


@cache_fn
def index_stream(fn, typ):
  assert typ in ("hevc", )

  with FileReader(fn) as f:
    assert os.path.exists(f.name), fn
    index, prefix = vidindex(f.name, typ)
    probe = ffprobe(f.name, typ)

  return {
    'index': index,
    'global_prefix': prefix,
    'probe': probe
  }


def index_videos(camera_paths, cache_prefix=None):
  """Requires that paths in camera_paths are contiguous and of the same type."""
  if len(camera_paths) < 1:
    raise ValueError("must provide at least one video to index")

  frame_type = fingerprint_video(camera_paths[0])
  for fn in camera_paths:
    index_video(fn, frame_type, cache_prefix)


def index_video(fn, frame_type=None, cache_prefix=None):
  cache_path = cache_path_for_file_path(fn, cache_prefix)

  if os.path.exists(cache_path):
    return

  if frame_type is None:
    frame_type = fingerprint_video(fn[0])

  if frame_type == FrameType.h265_stream:
    index_stream(fn, "hevc", cache_prefix=cache_prefix)
  else:
    raise NotImplementedError("Only h265 supported")


def get_video_index(fn, frame_type, cache_prefix=None):
  cache_path = cache_path_for_file_path(fn, cache_prefix)

  if not os.path.exists(cache_path):
    index_video(fn, frame_type, cache_prefix)

  if not os.path.exists(cache_path):
    return None
  with open(cache_path, "rb") as cache_file:
    return pickle.load(cache_file)


def read_file_check_size(f, sz, cookie):
  buff = bytearray(sz)
  bytes_read = f.readinto(buff)
  assert bytes_read == sz, (bytes_read, sz)
  return buff


def rgb24toyuv420(rgb):
  yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                           [-0.14714119, -0.28886916,  0.43601035 ],
                           [ 0.61497538, -0.51496512, -0.10001026 ]])
  img = np.dot(rgb.reshape(-1, 3), yuv_from_rgb.T).reshape(rgb.shape)

  y_len = img.shape[0] * img.shape[1]
  uv_len = y_len // 4

  ys = img[:, :, 0]
  us = (img[::2, ::2, 1] + img[1::2, ::2, 1] + img[::2, 1::2, 1] + img[1::2, 1::2, 1]) / 4 + 128
  vs = (img[::2, ::2, 2] + img[1::2, ::2, 2] + img[::2, 1::2, 2] + img[1::2, 1::2, 2]) / 4 + 128

  yuv420 = np.empty(y_len + 2 * uv_len, dtype=img.dtype)
  yuv420[:y_len] = ys.reshape(-1)
  yuv420[y_len:y_len + uv_len] = us.reshape(-1)
  yuv420[y_len + uv_len:y_len + 2 * uv_len] = vs.reshape(-1)

  return yuv420.clip(0, 255).astype('uint8')


def decompress_video_data(rawdat, vid_fmt, w, h, pix_fmt):
  # using a tempfile is much faster than proc.communicate for some reason

  with tempfile.TemporaryFile() as tmpf:
    tmpf.write(rawdat)
    tmpf.seek(0)

    threads = os.getenv("FFMPEG_THREADS", "0")
    cuda = os.getenv("FFMPEG_CUDA", "0") == "1"
    proc = subprocess.Popen(
      ["ffmpeg",
       "-threads", threads,
       "-hwaccel", "none" if not cuda else "cuda",
       "-c:v", "hevc",
       "-vsync", "0",
       "-f", vid_fmt,
       "-flags2", "showall",
       "-i", "pipe:0",
       "-threads", threads,
       "-f", "rawvideo",
       "-pix_fmt", pix_fmt,
       "pipe:1"],
      stdin=tmpf, stdout=subprocess.PIPE, stderr=open("/dev/null"))

    # dat = proc.communicate()[0]
    dat = proc.stdout.read()
    if proc.wait() != 0:
      raise DataUnreadableError("ffmpeg failed")

  if pix_fmt == "rgb24":
    ret = np.frombuffer(dat, dtype=np.uint8).reshape(-1, h, w, 3)
  elif pix_fmt == "yuv420p":
    ret = np.frombuffer(dat, dtype=np.uint8).reshape(-1, (h*w*3//2))
  elif pix_fmt == "yuv444p":
    ret = np.frombuffer(dat, dtype=np.uint8).reshape(-1, 3, h, w)
  else:
    raise NotImplementedError

  return ret


class BaseFrameReader:
  # properties: frame_type, frame_count, w, h

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()

  def close(self):
    pass

  def get(self, num, count=1, pix_fmt="yuv420p"):
    raise NotImplementedError


def FrameReader(fn, cache_prefix=None, readahead=False, readbehind=False, index_data=None):
  frame_type = fingerprint_video(fn)
  if frame_type == FrameType.raw:
    return RawFrameReader(fn)
  elif frame_type in (FrameType.h265_stream,):
    if not index_data:
      index_data = get_video_index(fn, frame_type, cache_prefix)
    return StreamFrameReader(fn, frame_type, index_data, readahead=readahead, readbehind=readbehind)
  else:
    raise NotImplementedError(frame_type)


class RawData:
  def __init__(self, f):
    self.f = _io.FileIO(f, 'rb')
    self.lenn = struct.unpack("I", self.f.read(4))[0]
    self.count = os.path.getsize(f) / (self.lenn+4)

  def read(self, i):
    self.f.seek((self.lenn+4)*i + 4)
    return self.f.read(self.lenn)


class RawFrameReader(BaseFrameReader):
  def __init__(self, fn):
    # raw camera
    self.fn = fn
    self.frame_type = FrameType.raw
    self.rawfile = RawData(self.fn)
    self.frame_count = self.rawfile.count
    self.w, self.h = 640, 480

  def load_and_debayer(self, img):
    img = np.frombuffer(img, dtype='uint8').reshape(960, 1280)
    cimg = np.dstack([img[0::2, 1::2], ((img[0::2, 0::2].astype("uint16") + img[1::2, 1::2].astype("uint16")) >> 1).astype("uint8"), img[1::2, 0::2]])
    return cimg

  def get(self, num, count=1, pix_fmt="yuv420p"):
    assert self.frame_count is not None
    assert num+count <= self.frame_count

    if pix_fmt not in ("yuv420p", "rgb24"):
      raise ValueError(f"Unsupported pixel format {pix_fmt!r}")

    app = []
    for i in range(num, num+count):
      dat = self.rawfile.read(i)
      rgb_dat = self.load_and_debayer(dat)
      if pix_fmt == "rgb24":
        app.append(rgb_dat)
      elif pix_fmt == "yuv420p":
        app.append(rgb24toyuv420(rgb_dat))
      else:
        raise NotImplementedError

    return app


class VideoStreamDecompressor:
  def __init__(self, fn, vid_fmt, w, h, pix_fmt):
    self.fn = fn
    self.vid_fmt = vid_fmt
    self.w = w
    self.h = h
    self.pix_fmt = pix_fmt

    if pix_fmt == "yuv420p":
      self.out_size = w*h*3//2  # yuv420p
    elif pix_fmt in ("rgb24", "yuv444p"):
      self.out_size = w*h*3
    else:
      raise NotImplementedError

    self.proc = None
    self.t = threading.Thread(target=self.write_thread)
    self.t.daemon = True

  def write_thread(self):
    try:
      with FileReader(self.fn) as f:
        while True:
          r = f.read(1024*1024)
          if len(r) == 0:
            break
          self.proc.stdin.write(r)
    finally:
      self.proc.stdin.close()

  def read(self):
    threads = os.getenv("FFMPEG_THREADS", "0")
    cuda = os.getenv("FFMPEG_CUDA", "0") == "1"
    cmd = [
      "ffmpeg",
      "-threads", threads,
      "-hwaccel", "none" if not cuda else "cuda",
      "-c:v", "hevc",
      # "-avioflags", "direct",
      "-analyzeduration", "0",
      "-probesize", "32",
      "-flush_packets", "0",
      # "-fflags", "nobuffer",
      "-vsync", "0",
      "-f", self.vid_fmt,
      "-i", "pipe:0",
      "-threads", threads,
      "-f", "rawvideo",
      "-pix_fmt", self.pix_fmt,
      "pipe:1"
    ]
    self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
      self.t.start()

      while True:
        dat = self.proc.stdout.read(self.out_size)
        if len(dat) == 0:
          break
        assert len(dat) == self.out_size
        if self.pix_fmt == "rgb24":
          ret = np.frombuffer(dat, dtype=np.uint8).reshape((self.h, self.w, 3))
        elif self.pix_fmt == "yuv420p":
          ret = np.frombuffer(dat, dtype=np.uint8)
        elif self.pix_fmt == "yuv444p":
          ret = np.frombuffer(dat, dtype=np.uint8).reshape((3, self.h, self.w))
        else:
          assert False
        yield ret

      result_code = self.proc.wait()
      assert result_code == 0, result_code
    finally:
      self.proc.kill()
      self.t.join()

class StreamGOPReader(GOPReader):
  def __init__(self, fn, frame_type, index_data):
    assert frame_type == FrameType.h265_stream

    self.fn = fn

    self.frame_type = frame_type
    self.frame_count = None
    self.w, self.h = None, None

    self.prefix = None
    self.index = None

    self.index = index_data['index']
    self.prefix = index_data['global_prefix']
    probe = index_data['probe']

    self.prefix_frame_data = None
    self.num_prefix_frames = 0
    self.vid_fmt = "hevc"

    i = 0
    while i < self.index.shape[0] and self.index[i, 0] != HEVC_SLICE_I:
      i += 1
    self.first_iframe = i

    assert self.first_iframe == 0

    self.frame_count = len(self.index) - 1

    self.w = probe['streams'][0]['width']
    self.h = probe['streams'][0]['height']

  def _lookup_gop(self, num):
    frame_b = num
    while frame_b > 0 and self.index[frame_b, 0] != HEVC_SLICE_I:
      frame_b -= 1

    frame_e = num + 1
    while frame_e < (len(self.index) - 1) and self.index[frame_e, 0] != HEVC_SLICE_I:
      frame_e += 1

    offset_b = self.index[frame_b, 1]
    offset_e = self.index[frame_e, 1]

    return (frame_b, frame_e, offset_b, offset_e)

  def get_gop(self, num):
    frame_b, frame_e, offset_b, offset_e = self._lookup_gop(num)
    assert frame_b <= num < frame_e

    num_frames = frame_e - frame_b

    with FileReader(self.fn) as f:
      f.seek(offset_b)
      rawdat = f.read(offset_e - offset_b)

      if num < self.first_iframe:
        assert self.prefix_frame_data
        rawdat = self.prefix_frame_data + rawdat

      rawdat = self.prefix + rawdat

    skip_frames = 0
    if num < self.first_iframe:
      skip_frames = self.num_prefix_frames

    return frame_b, num_frames, skip_frames, rawdat


class GOPFrameReader(BaseFrameReader):
  #FrameReader with caching and readahead for formats that are group-of-picture based

  def __init__(self, readahead=False, readbehind=False):
    self.open_ = True

    self.readahead = readahead
    self.readbehind = readbehind
    self.frame_cache = LRU(64)

    if self.readahead:
      self.cache_lock = threading.RLock()
      self.readahead_last = None
      self.readahead_len = 30
      self.readahead_c = threading.Condition()
      self.readahead_thread = threading.Thread(target=self._readahead_thread)
      self.readahead_thread.daemon = True
      self.readahead_thread.start()
    else:
      self.cache_lock = DoNothingContextManager()

  def close(self):
    if not self.open_:
      return
    self.open_ = False

    if self.readahead:
      self.readahead_c.acquire()
      self.readahead_c.notify()
      self.readahead_c.release()
      self.readahead_thread.join()

  def _readahead_thread(self):
    while True:
      self.readahead_c.acquire()
      try:
        if not self.open_:
          break
        self.readahead_c.wait()
      finally:
        self.readahead_c.release()
      if not self.open_:
        break
      assert self.readahead_last
      num, pix_fmt = self.readahead_last

      if self.readbehind:
        for k in range(num - 1, max(0, num - self.readahead_len), -1):
          self._get_one(k, pix_fmt)
      else:
        for k in range(num, min(self.frame_count, num + self.readahead_len)):
          self._get_one(k, pix_fmt)

  def _get_one(self, num, pix_fmt):
    assert num < self.frame_count

    if (num, pix_fmt) in self.frame_cache:
      return self.frame_cache[(num, pix_fmt)]

    with self.cache_lock:
      if (num, pix_fmt) in self.frame_cache:
        return self.frame_cache[(num, pix_fmt)]

      frame_b, num_frames, skip_frames, rawdat = self.get_gop(num)

      ret = decompress_video_data(rawdat, self.vid_fmt, self.w, self.h, pix_fmt)
      ret = ret[skip_frames:]
      assert ret.shape[0] == num_frames

      for i in range(ret.shape[0]):
        self.frame_cache[(frame_b+i, pix_fmt)] = ret[i]

      return self.frame_cache[(num, pix_fmt)]

  def get(self, num, count=1, pix_fmt="yuv420p"):
    assert self.frame_count is not None

    if num + count > self.frame_count:
      raise ValueError(f"{num + count} > {self.frame_count}")

    if pix_fmt not in ("yuv420p", "rgb24", "yuv444p"):
      raise ValueError(f"Unsupported pixel format {pix_fmt!r}")

    ret = [self._get_one(num + i, pix_fmt) for i in range(count)]

    if self.readahead:
      self.readahead_last = (num+count, pix_fmt)
      self.readahead_c.acquire()
      self.readahead_c.notify()
      self.readahead_c.release()

    return ret


class StreamFrameReader(StreamGOPReader, GOPFrameReader):
  def __init__(self, fn, frame_type, index_data, readahead=False, readbehind=False):
    StreamGOPReader.__init__(self, fn, frame_type, index_data)
    GOPFrameReader.__init__(self, readahead, readbehind)


def GOPFrameIterator(gop_reader, pix_fmt):
  dec = VideoStreamDecompressor(gop_reader.fn, gop_reader.vid_fmt, gop_reader.w, gop_reader.h, pix_fmt)
  for frame in dec.read():
    yield frame


def FrameIterator(fn, pix_fmt, **kwargs):
  fr = FrameReader(fn, **kwargs)
  if isinstance(fr, GOPReader):
    for v in GOPFrameIterator(fr, pix_fmt):
      yield v
  else:
    for i in range(fr.frame_count):
      yield fr.get(i, pix_fmt=pix_fmt)[0]
