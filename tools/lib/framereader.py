import os
import sys
import json
import struct
import tempfile
import threading
import xml.etree.ElementTree as ET
import numpy as np
if sys.version_info >= (3,0):
  import queue
  import pickle
  from io import BytesIO as StringIO
else:
  import Queue as queue
  import cPickle as pickle
  from cStringIO import StringIO

import subprocess
from aenum import Enum
from lru import LRU
from functools import wraps

from tools.lib.cache import cache_path_for_file_path
from tools.lib.exceptions import DataUnreadableError
try:
  from xx.chffr.lib.filereader import FileReader
except ImportError:
  from tools.lib.filereader import FileReader
from tools.lib.file_helpers import atomic_write_in_dir
from tools.lib.mkvparse import mkvindex
from tools.lib.route import Route

H264_SLICE_P = 0
H264_SLICE_B = 1
H264_SLICE_I = 2

HEVC_SLICE_B = 0
HEVC_SLICE_P = 1
HEVC_SLICE_I = 2

SLICE_I = 2 # hevc and h264 are the same :)

class FrameType(Enum):
  raw = 1
  h265_stream = 2
  h264_mp4 = 3
  h264_pstream = 4
  ffv1_mkv = 5
  ffvhuff_mkv = 6

def fingerprint_video(fn):
  with FileReader(fn) as f:
    header = f.read(4)
  if len(header) == 0:
    raise DataUnreadableError("%s is empty" % fn)
  elif header == b"\x00\xc0\x12\x00":
    return FrameType.raw
  elif header == b"\x00\x00\x00\x01":
    if 'hevc' in fn:
      return FrameType.h265_stream
    elif os.path.basename(fn) in ("camera", "acamera"):
      return FrameType.h264_pstream
    else:
      raise NotImplementedError(fn)
  elif header == b"\x00\x00\x00\x1c":
    return FrameType.h264_mp4
  elif header == b"\x1a\x45\xdf\xa3":
    return FrameType.ffv1_mkv
  else:
    raise NotImplementedError(fn)


def ffprobe(fn, fmt=None):
  cmd = ["ffprobe",
    "-v", "quiet",
    "-print_format", "json",
    "-show_format", "-show_streams"]
  if fmt:
    cmd += ["-format", fmt]
  cmd += [fn]

  try:
    ffprobe_output = subprocess.check_output(cmd)
  except subprocess.CalledProcessError as e:
    raise DataUnreadableError(fn)

  return json.loads(ffprobe_output)


def vidindex(fn, typ):
  vidindex_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vidindex")
  vidindex = os.path.join(vidindex_dir, "vidindex")

  subprocess.check_call(["make"], cwd=vidindex_dir, stdout=open("/dev/null","w"))

  with tempfile.NamedTemporaryFile() as prefix_f, \
       tempfile.NamedTemporaryFile() as index_f:
    try:
      subprocess.check_call([vidindex, typ, fn, prefix_f.name, index_f.name])
    except subprocess.CalledProcessError as e:
      raise DataUnreadableError("vidindex failed on file %s" % fn)
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
  assert typ in ("hevc", "h264")

  with FileReader(fn) as f:
    assert os.path.exists(f.name), fn
    index, prefix = vidindex(f.name, typ)
    probe = ffprobe(f.name, typ)

  return {
    'index': index,
    'global_prefix': prefix,
    'probe': probe
  }

@cache_fn
def index_mp4(fn):
  with FileReader(fn) as f:
    return vidindex_mp4(f.name)

@cache_fn
def index_mkv(fn):
  with FileReader(fn) as f:
    probe = ffprobe(f.name, "matroska")
    with open(f.name, "rb") as d_f:
      config_record, index = mkvindex.mkvindex(d_f)
  return {
    'probe': probe,
    'config_record': config_record,
    'index': index
  }

def index_videos(camera_paths, cache_prefix=None):
  """Requires that paths in camera_paths are contiguous and of the same type."""
  if len(camera_paths) < 1:
    raise ValueError("must provide at least one video to index")

  frame_type = fingerprint_video(camera_paths[0])
  if frame_type == FrameType.h264_pstream:
    index_pstream(camera_paths, "h264", cache_prefix)
  else:
    for fn in camera_paths:
      index_video(fn, frame_type, cache_prefix)

def index_video(fn, frame_type=None, cache_prefix=None):
  cache_path = cache_path_for_file_path(fn, cache_prefix)

  if os.path.exists(cache_path):
    return

  if frame_type is None:
    frame_type = fingerprint_video(fn[0])

  if frame_type == FrameType.h264_pstream:
    #hack: try to index the whole route now
    route = Route.from_file_path(fn)

    camera_paths = route.camera_paths()
    if fn not in camera_paths:
      raise DataUnreadableError("Not a contiguous route camera file: {}".format(fn))

    print("no pstream cache for %s, indexing route %s now" % (fn, route.name))
    index_pstream(route.camera_paths(), "h264", cache_prefix)
  elif frame_type == FrameType.h265_stream:
    index_stream(fn, "hevc", cache_prefix=cache_prefix)
  elif frame_type == FrameType.h264_mp4:
    index_mp4(fn, cache_prefix=cache_prefix)

def get_video_index(fn, frame_type, cache_prefix=None):
  cache_path = cache_path_for_file_path(fn, cache_prefix)

  if not os.path.exists(cache_path):
    index_video(fn, frame_type, cache_prefix)

  if not os.path.exists(cache_path):
    return None
  with open(cache_path, "rb") as cache_file:
    return pickle.load(cache_file)

def pstream_predecompress(fns, probe, indexes, global_prefix, cache_prefix, multithreaded=False):
  assert len(fns) == len(indexes)
  out_fns = [cache_path_for_file_path(fn, cache_prefix, extension=".predecom.mkv") for fn in fns]
  out_exists = map(os.path.exists, out_fns)
  if all(out_exists):
    return

  w = probe['streams'][0]['width']
  h = probe['streams'][0]['height']

  frame_size = w*h*3/2 # yuv420p

  decompress_proc = subprocess.Popen(
    ["ffmpeg",
     "-threads", "0" if multithreaded else "1",
     "-vsync", "0",
     "-f", "h264",
     "-i", "pipe:0",
     "-threads", "0" if multithreaded else "1",
     "-f", "rawvideo",
     "-pix_fmt", "yuv420p",
     "pipe:1"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/dev/null", "wb"))

  def write_thread():
    for fn in fns:
      with FileReader(fn) as f:
        decompress_proc.stdin.write(f.read())
    decompress_proc.stdin.close()

  def read_frame():
    frame = None
    try:
      frame = decompress_proc.stdout.read(frame_size)
    except (IOError, ValueError):
      pass
    if frame is None or frame == "" or len(frame) != frame_size:
      raise DataUnreadableError("pre-decompression failed for %s" % fn)
    return frame

  t = threading.Thread(target=write_thread)
  t.daemon = True
  t.start()

  try:
    for fn, out_fn, out_exist, index in zip(fns, out_fns, out_exists, indexes):
      if out_exist:
        for fi in range(index.shape[0]-1):
          read_frame()
        continue

      with atomic_write_in_dir(out_fn, mode="w+b", overwrite=True) as out_tmp:
        compress_proc = subprocess.Popen(
          ["ffmpeg",
          "-threads", "0" if multithreaded else "1",
           "-y",
           "-vsync", "0",
           "-f", "rawvideo",
           "-pix_fmt", "yuv420p",
           "-s", "%dx%d" % (w, h),
           "-i", "pipe:0",
          "-threads", "0" if multithreaded else "1",
           "-f", "matroska",
           "-vcodec", "ffv1",
           "-g", "0",
           out_tmp.name],
          stdin=subprocess.PIPE, stderr=open("/dev/null", "wb"))
        try:
          for fi in range(index.shape[0]-1):
            frame = read_frame()
            compress_proc.stdin.write(frame)
          compress_proc.stdin.close()
        except:
          compress_proc.kill()
          raise

        assert compress_proc.wait() == 0

      cache_path = cache_path_for_file_path(fn, cache_prefix)
      with atomic_write_in_dir(cache_path, mode="wb", overwrite=True) as cache_file:
        pickle.dump({
          'predecom': os.path.basename(out_fn),
          'index': index,
          'probe': probe,
          'global_prefix': global_prefix,
        }, cache_file, -1)

  except:
    decompress_proc.kill()
    raise
  finally:
    t.join()

  rc = decompress_proc.wait()
  if rc != 0:
    raise DataUnreadableError(fns[0])


def index_pstream(fns, typ, cache_prefix=None):
  if typ != "h264":
    raise NotImplementedError(typ)

  if not fns:
    raise DataUnreadableError("chffr h264 requires contiguous files")

  out_fns = [cache_path_for_file_path(fn, cache_prefix) for fn in fns]
  out_exists = map(os.path.exists, out_fns)
  if all(out_exists): return

  # load existing index files to avoid re-doing work
  existing_indexes = []
  for out_fn, exists in zip(out_fns, out_exists):
    existing = None
    if exists:
      with open(out_fn, "rb") as cache_file:
        existing = pickle.load(cache_file)
    existing_indexes.append(existing)

  # probe the first file
  if existing_indexes[0]:
    probe = existing_indexes[0]['probe']
  else:
    with FileReader(fns[0]) as f:
      probe = ffprobe(f.name, typ)

  global_prefix = None

  # get the video index of all the segments in this stream
  indexes = []
  for i, fn in enumerate(fns):
    if existing_indexes[i]:
      index = existing_indexes[i]['index']
      prefix = existing_indexes[i]['global_prefix']
    else:
      with FileReader(fn) as f:
        index, prefix = vidindex(f.name, typ)
    if i == 0:
      # assert prefix
      if not prefix:
        raise DataUnreadableError("vidindex failed for %s" % fn)
      global_prefix = prefix
    indexes.append(index)

  assert global_prefix

  if np.sum(indexes[0][:, 0] == H264_SLICE_I) <= 1:
    print("pstream %s is unseekable. pre-decompressing all the segments..." % (fns[0]))
    pstream_predecompress(fns, probe, indexes, global_prefix, cache_prefix)
    return

  # generate what's required to make each segment self-contained
  # (the partial GOP from the end of each segments are put asside to add
  #  to the start of the following segment)
  prefix_data = ["" for _ in fns]
  prefix_index = [[] for _ in fns]
  for i in range(len(fns)-1):
    if indexes[i+1][0, 0] == H264_SLICE_I and indexes[i+1][0, 1] <= 1:
      # next file happens to start with a i-frame, dont need use this file's end
      continue

    index = indexes[i]
    if i == 0 and np.sum(index[:, 0] == H264_SLICE_I) <= 1:
      raise NotImplementedError("No I-frames in pstream.")

    # find the last GOP in the index
    frame_b = len(index)-1
    while frame_b > 0 and index[frame_b, 0] != H264_SLICE_I:
      frame_b -= 1

    assert frame_b >= 0
    assert index[frame_b, 0] == H264_SLICE_I

    end_len = len(index)-frame_b

    with FileReader(fns[i]) as vid:
      vid.seek(index[frame_b, 1])
      end_data = vid.read()

    prefix_data[i+1] = end_data
    prefix_index[i+1] = index[frame_b:-1]
    # indexes[i] = index[:frame_b]

  for i, fn in enumerate(fns):
    cache_path = out_fns[i]

    if os.path.exists(cache_path):
      continue

    segment_index = {
      'index': indexes[i],
      'global_prefix': global_prefix,
      'probe': probe,
      'prefix_frame_data': prefix_data[i], # data to prefix the first GOP with
      'num_prefix_frames': len(prefix_index[i]), # number of frames to skip in the first GOP
    }

    with atomic_write_in_dir(cache_path, mode="wb", overwrite=True) as cache_file:
      pickle.dump(segment_index, cache_file, -1)

def read_file_check_size(f, sz, cookie):
  buff = bytearray(sz)
  bytes_read = f.readinto(buff)
  assert bytes_read == sz, (bytes_read, sz)
  return buff


import signal
import ctypes
def _set_pdeathsig(sig=signal.SIGTERM):
  def f():
    libc = ctypes.CDLL('libc.so.6')
    return libc.prctl(1, sig)
  return f

def vidindex_mp4(fn):
  try:
    xmls = subprocess.check_output(["MP4Box", fn, "-diso", "-out", "/dev/stdout"])
  except subprocess.CalledProcessError as e:
    raise DataUnreadableError(fn)

  tree = ET.fromstring(xmls)

  def parse_content(s):
    assert s.startswith("data:application/octet-string,")
    return s[len("data:application/octet-string,"):].decode("hex")

  avc_element = tree.find(".//AVCSampleEntryBox")
  width = int(avc_element.attrib['Width'])
  height = int(avc_element.attrib['Height'])

  sps_element = avc_element.find(".//AVCDecoderConfigurationRecord/SequenceParameterSet")
  pps_element = avc_element.find(".//AVCDecoderConfigurationRecord/PictureParameterSet")

  sps = parse_content(sps_element.attrib['content'])
  pps = parse_content(pps_element.attrib['content'])

  media_header = tree.find("MovieBox/TrackBox/MediaBox/MediaHeaderBox")
  time_scale = int(media_header.attrib['TimeScale'])

  sample_sizes = [
    int(entry.attrib['Size']) for entry in tree.findall(
      "MovieBox/TrackBox/MediaBox/MediaInformationBox/SampleTableBox/SampleSizeBox/SampleSizeEntry")
  ]

  sample_dependency = [
    entry.attrib['dependsOnOther'] == "yes" for entry in tree.findall(
      "MovieBox/TrackBox/MediaBox/MediaInformationBox/SampleTableBox/SampleDependencyTypeBox/SampleDependencyEntry")
  ]

  assert len(sample_sizes) == len(sample_dependency)

  chunk_offsets = [
    int(entry.attrib['offset']) for entry in tree.findall(
      "MovieBox/TrackBox/MediaBox/MediaInformationBox/SampleTableBox/ChunkOffsetBox/ChunkEntry")
  ]

  sample_chunk_table = [
    (int(entry.attrib['FirstChunk'])-1, int(entry.attrib['SamplesPerChunk'])) for entry in tree.findall(
      "MovieBox/TrackBox/MediaBox/MediaInformationBox/SampleTableBox/SampleToChunkBox/SampleToChunkEntry")
  ]

  sample_offsets = [None for _ in sample_sizes]

  sample_i = 0
  for i, (first_chunk, samples_per_chunk) in enumerate(sample_chunk_table):
    if i == len(sample_chunk_table)-1:
      last_chunk = len(chunk_offsets)-1
    else:
      last_chunk = sample_chunk_table[i+1][0]-1
    for k in range(first_chunk, last_chunk+1):
      sample_offset = chunk_offsets[k]
      for _ in range(samples_per_chunk):
        sample_offsets[sample_i] = sample_offset
        sample_offset += sample_sizes[sample_i]
        sample_i += 1

  assert sample_i == len(sample_sizes)

  pts_offset_table = [
    ( int(entry.attrib['CompositionOffset']), int(entry.attrib['SampleCount']) ) for entry in tree.findall(
      "MovieBox/TrackBox/MediaBox/MediaInformationBox/SampleTableBox/CompositionOffsetBox/CompositionOffsetEntry")
  ]
  sample_pts_offset = [0 for _ in sample_sizes]
  sample_i = 0
  for dt, count in pts_offset_table:
    for _ in range(count):
      sample_pts_offset[sample_i] = dt
      sample_i += 1

  sample_time_table = [
    ( int(entry.attrib['SampleDelta']), int(entry.attrib['SampleCount']) ) for entry in tree.findall(
      "MovieBox/TrackBox/MediaBox/MediaInformationBox/SampleTableBox/TimeToSampleBox/TimeToSampleEntry")
  ]
  sample_time = [None for _ in sample_sizes]
  cur_ts = 0
  sample_i = 0
  for dt, count in sample_time_table:
    for _ in range(count):
      sample_time[sample_i] = (cur_ts + sample_pts_offset[sample_i]) * 1000 / time_scale

      cur_ts += dt
      sample_i += 1

  sample_time.sort() # because we ony decode GOPs in PTS order

  return {
    'width': width,
    'height': height,
    'sample_offsets': sample_offsets,
    'sample_sizes': sample_sizes,
    'sample_dependency': sample_dependency,
    'sample_time': sample_time,
    'sps': sps,
    'pps': pps
  }


class BaseFrameReader(object):
  # properties: frame_type, frame_count, w, h

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()

  def close(self):
    pass

  def get(self, num, count=1, pix_fmt="yuv420p"):
    raise NotImplementedError

def FrameReader(fn, cache_prefix=None, readahead=False, readbehind=False, multithreaded=True, index_data=None):
  frame_type = fingerprint_video(fn)
  if frame_type == FrameType.raw:
    return RawFrameReader(fn)
  elif frame_type in (FrameType.h265_stream, FrameType.h264_pstream):
    if not index_data:
      index_data = get_video_index(fn, frame_type, cache_prefix)
    if index_data is not None and "predecom" in index_data:
      cache_path = cache_path_for_file_path(fn, cache_prefix)
      return MKVFrameReader(
        os.path.join(os.path.dirname(cache_path), index_data["predecom"]))
    else:
      return StreamFrameReader(fn, frame_type, index_data,
        readahead=readahead, readbehind=readbehind, multithreaded=multithreaded)
  elif frame_type == FrameType.h264_mp4:
    return MP4FrameReader(fn, readahead=readahead)
  elif frame_type == FrameType.ffv1_mkv:
    return MKVFrameReader(fn)
  else:
    raise NotImplementedError(frame_type)

def rgb24toyuv420(rgb):
  yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                           [-0.14714119, -0.28886916,  0.43601035 ],
                           [ 0.61497538, -0.51496512, -0.10001026 ]])
  img = np.dot(rgb.reshape(-1, 3), yuv_from_rgb.T).reshape(rgb.shape)

  y_len = img.shape[0] * img.shape[1]
  uv_len = y_len / 4

  ys = img[:, :, 0]
  us = (img[::2, ::2, 1] + img[1::2, ::2, 1] + img[::2, 1::2, 1] + img[1::2, 1::2, 1]) / 4 + 128
  vs = (img[::2, ::2, 2] + img[1::2, ::2, 2] + img[::2, 1::2, 2] + img[1::2, 1::2, 2]) / 4 + 128

  yuv420 = np.empty(y_len + 2 * uv_len, dtype=img.dtype)
  yuv420[:y_len] = ys.reshape(-1)
  yuv420[y_len:y_len + uv_len] = us.reshape(-1)
  yuv420[y_len + uv_len:y_len + 2 * uv_len] = vs.reshape(-1)

  return yuv420.clip(0,255).astype('uint8')

class RawData(object):
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
    cimg = np.dstack([img[0::2, 1::2], (
      (img[0::2, 0::2].astype("uint16") + img[1::2, 1::2].astype("uint16"))
      >> 1).astype("uint8"), img[1::2, 0::2]])
    return cimg


  def get(self, num, count=1, pix_fmt="yuv420p"):
    assert self.frame_count is not None
    assert num+count <= self.frame_count

    if pix_fmt not in ("yuv420p", "rgb24"):
      raise ValueError("Unsupported pixel format %r" % pix_fmt)

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

def decompress_video_data(rawdat, vid_fmt, w, h, pix_fmt, multithreaded=False):
  # using a tempfile is much faster than proc.communicate for some reason

  with tempfile.TemporaryFile() as tmpf:
    tmpf.write(rawdat)
    tmpf.seek(0)

    proc = subprocess.Popen(
      ["ffmpeg",
       "-threads", "0" if multithreaded else "1",
       "-vsync", "0",
       "-f", vid_fmt,
       "-flags2", "showall",
       "-i", "pipe:0",
       "-threads", "0" if multithreaded else "1",
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

class VideoStreamDecompressor(object):
  def __init__(self, vid_fmt, w, h, pix_fmt, multithreaded=False):
    self.vid_fmt = vid_fmt
    self.w = w
    self.h = h
    self.pix_fmt = pix_fmt

    if pix_fmt == "yuv420p":
      self.out_size = w*h*3//2 # yuv420p
    elif pix_fmt in ("rgb24", "yuv444p"):
      self.out_size = w*h*3
    else:
      raise NotImplementedError

    self.out_q = queue.Queue()

    self.proc = subprocess.Popen(
      ["ffmpeg",
       "-threads", "0" if multithreaded else "1",
       # "-avioflags", "direct",
       "-analyzeduration", "0",
       "-probesize", "32",
       "-flush_packets", "0",
       # "-fflags", "nobuffer",
       "-vsync", "0",
       "-f", vid_fmt,
       "-i", "pipe:0",
       "-threads", "0" if multithreaded else "1",
       "-f", "rawvideo",
       "-pix_fmt", pix_fmt,
       "pipe:1"],
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/dev/null", "wb"))

    def read_thread():
      while True:
        r = self.proc.stdout.read(self.out_size)
        if len(r) == 0:
          break
        assert len(r) == self.out_size
        self.out_q.put(r)

    self.t = threading.Thread(target=read_thread)
    self.t.daemon = True
    self.t.start()

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()

  def write(self, rawdat):
    self.proc.stdin.write(rawdat)
    self.proc.stdin.flush()

  def read(self):
    dat = self.out_q.get(block=True)

    if self.pix_fmt == "rgb24":
      ret = np.frombuffer(dat, dtype=np.uint8).reshape((self.h, self.w, 3))
    elif self.pix_fmt == "yuv420p":
      ret = np.frombuffer(dat, dtype=np.uint8)
    elif self.pix_fmt == "yuv444p":
      ret = np.frombuffer(dat, dtype=np.uint8).reshape((3, self.h, self.w))
    else:
      assert False

    return ret

  def eos(self):
    self.proc.stdin.close()

  def close(self):
    self.proc.stdin.close()
    self.t.join()
    self.proc.wait()
    assert self.proc.wait() == 0


class MKVFrameReader(BaseFrameReader):
  def __init__(self, fn):
    self.fn = fn

    #print("MKVFrameReader", fn)
    index_data = index_mkv(fn)
    stream = index_data['probe']['streams'][0]
    self.w = stream['width']
    self.h = stream['height']

    if stream['codec_name'] == 'ffv1':
      self.frame_type = FrameType.ffv1_mkv
    elif stream['codec_name'] == 'ffvhuff':
      self.frame_type = FrameType.ffvhuff_mkv
    else:
      raise NotImplementedError

    self.config_record = index_data['config_record']
    self.index = index_data['index']

    self.frame_count = len(self.index)

  def get(self, num, count=1, pix_fmt="yuv420p"):
    assert 0 < num+count <= self.frame_count

    frame_dats = []
    with FileReader(self.fn) as f:
      for i in range(num, num+count):
        pos, length, _ = self.index[i]
        f.seek(pos)
        frame_dats.append(f.read(length))

    of = StringIO()
    mkvindex.simple_gen(of, self.config_record, self.w, self.h, frame_dats)

    r = decompress_video_data(of.getvalue(), "matroska", self.w, self.h, pix_fmt)
    assert len(r) == count

    return r


class GOPReader(object):
  def get_gop(self, num):
    # returns (start_frame_num, num_frames, frames_to_skip, gop_data)
    raise NotImplementedError


class DoNothingContextManager(object):
  def __enter__(self): return self
  def __exit__(*x): pass


class GOPFrameReader(BaseFrameReader):
  #FrameReader with caching and readahead for formats that are group-of-picture based

  def __init__(self, readahead=False, readbehind=False, multithreaded=True):
    self.open_ = True

    self.multithreaded = multithreaded
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
        for k in range(num-1, max(0, num-self.readahead_len), -1):
          self._get_one(k, pix_fmt)
      else:
        for k in range(num, min(self.frame_count, num+self.readahead_len)):
          self._get_one(k, pix_fmt)

  def _get_one(self, num, pix_fmt):
    assert num < self.frame_count

    if (num, pix_fmt) in self.frame_cache:
      return self.frame_cache[(num, pix_fmt)]

    with self.cache_lock:
      if (num, pix_fmt) in self.frame_cache:
        return self.frame_cache[(num, pix_fmt)]

      frame_b, num_frames, skip_frames, rawdat = self.get_gop(num)

      ret = decompress_video_data(rawdat, self.vid_fmt, self.w, self.h, pix_fmt,
                                  multithreaded=self.multithreaded)
      ret = ret[skip_frames:]
      assert ret.shape[0] == num_frames

      for i in range(ret.shape[0]):
        self.frame_cache[(frame_b+i, pix_fmt)] = ret[i]

      return self.frame_cache[(num, pix_fmt)]

  def get(self, num, count=1, pix_fmt="yuv420p"):
    assert self.frame_count is not None

    if num + count > self.frame_count:
      raise ValueError("{} > {}".format(num + count, self.frame_count))

    if pix_fmt not in ("yuv420p", "rgb24", "yuv444p"):
      raise ValueError("Unsupported pixel format %r" % pix_fmt)

    ret = [self._get_one(num + i, pix_fmt) for i in range(count)]

    if self.readahead:
      self.readahead_last = (num+count, pix_fmt)
      self.readahead_c.acquire()
      self.readahead_c.notify()
      self.readahead_c.release()

    return ret

class MP4GOPReader(GOPReader):
  def __init__(self, fn):
    self.fn = fn
    self.frame_type = FrameType.h264_mp4

    self.index = index_mp4(fn)

    self.w = self.index['width']
    self.h = self.index['height']
    self.sample_sizes = self.index['sample_sizes']
    self.sample_offsets = self.index['sample_offsets']
    self.sample_dependency = self.index['sample_dependency']

    self.vid_fmt = "h264"

    self.frame_count = len(self.sample_sizes)

    self.prefix = "\x00\x00\x00\x01"+self.index['sps']+"\x00\x00\x00\x01"+self.index['pps']

  def _lookup_gop(self, num):
    frame_b = num
    while frame_b > 0 and self.sample_dependency[frame_b]:
      frame_b -= 1

    frame_e = num+1
    while frame_e < (len(self.sample_dependency)-1) and self.sample_dependency[frame_e]:
      frame_e += 1

    return (frame_b, frame_e)

  def get_gop(self, num):
    frame_b, frame_e = self._lookup_gop(num)
    assert frame_b <= num < frame_e

    num_frames = frame_e-frame_b

    with FileReader(self.fn) as f:
      rawdat = []

      sample_i = frame_b
      while sample_i < frame_e:
        size = self.sample_sizes[sample_i]
        start_offset = self.sample_offsets[sample_i]

        # try to read contiguously because a read could actually be a http request
        sample_i += 1
        while sample_i < frame_e and size < 10000000 and start_offset+size == self.sample_offsets[sample_i]:
          size += self.sample_sizes[sample_i]
          sample_i += 1

        f.seek(start_offset)
        sampledat = f.read(size)

        # read length-prefixed NALUs and output in Annex-B
        i = 0
        while i < len(sampledat):
          nal_len, = struct.unpack(">I", sampledat[i:i+4])
          rawdat.append("\x00\x00\x00\x01"+sampledat[i+4:i+4+nal_len])
          i = i+4+nal_len
        assert i == len(sampledat)

    rawdat = self.prefix+''.join(rawdat)

    return frame_b, num_frames, 0, rawdat

class MP4FrameReader(MP4GOPReader, GOPFrameReader):
  def __init__(self, fn, readahead=False):
    MP4GOPReader.__init__(self, fn)
    GOPFrameReader.__init__(self, readahead)

class StreamGOPReader(GOPReader):
  def __init__(self, fn, frame_type, index_data):
    self.fn = fn

    self.frame_type = frame_type
    self.frame_count = None
    self.w, self.h = None, None

    self.prefix = None
    self.index = None

    self.index = index_data['index']
    self.prefix = index_data['global_prefix']
    probe = index_data['probe']

    if self.frame_type == FrameType.h265_stream:
      self.prefix_frame_data = None
      self.num_prefix_frames = 0
      self.vid_fmt = "hevc"

    elif self.frame_type == FrameType.h264_pstream:
      self.prefix_frame_data = index_data['prefix_frame_data']
      self.num_prefix_frames = index_data['num_prefix_frames']

      self.vid_fmt = "h264"

    i = 0
    while i < self.index.shape[0] and self.index[i, 0] != SLICE_I:
      i += 1
    self.first_iframe = i

    if self.frame_type == FrameType.h265_stream:
      assert self.first_iframe == 0

    self.frame_count = len(self.index)-1

    self.w = probe['streams'][0]['width']
    self.h = probe['streams'][0]['height']


  def _lookup_gop(self, num):
    frame_b = num
    while frame_b > 0 and self.index[frame_b, 0] != SLICE_I:
      frame_b -= 1

    frame_e = num+1
    while frame_e < (len(self.index)-1) and self.index[frame_e, 0] != SLICE_I:
      frame_e += 1

    offset_b = self.index[frame_b, 1]
    offset_e = self.index[frame_e, 1]

    return (frame_b, frame_e, offset_b, offset_e)

  def get_gop(self, num):
    frame_b, frame_e, offset_b, offset_e = self._lookup_gop(num)
    assert frame_b <= num < frame_e

    num_frames = frame_e-frame_b

    with FileReader(self.fn) as f:
      f.seek(offset_b)
      rawdat = f.read(offset_e-offset_b)

      if num < self.first_iframe:
        assert self.prefix_frame_data
        rawdat = self.prefix_frame_data + rawdat

      rawdat = self.prefix + rawdat

    skip_frames = 0
    if num < self.first_iframe:
      skip_frames = self.num_prefix_frames

    return frame_b, num_frames, skip_frames, rawdat

class StreamFrameReader(StreamGOPReader, GOPFrameReader):
  def __init__(self, fn, frame_type, index_data, readahead=False, readbehind=False, multithreaded=False):
    StreamGOPReader.__init__(self, fn, frame_type, index_data)
    GOPFrameReader.__init__(self, readahead, readbehind, multithreaded)




def GOPFrameIterator(gop_reader, pix_fmt, multithreaded=True):
  # this is really ugly. ill think about how to refactor it when i can think good

  IN_FLIGHT_GOPS = 6 # should be enough that the stream decompressor starts returning data

  with VideoStreamDecompressor(
      gop_reader.vid_fmt, gop_reader.w, gop_reader.h, pix_fmt, multithreaded) as dec:

    read_work = []

    def readthing():
      # print read_work, dec.out_q.qsize()
      outf = dec.read()
      read_thing = read_work[0]
      if read_thing[0] > 0:
        read_thing[0] -= 1
      else:
        assert read_thing[1] > 0
        yield outf
        read_thing[1] -= 1

      if read_thing[1] == 0:
        read_work.pop(0)

    i = 0
    while i < gop_reader.frame_count:
      frame_b, num_frames, skip_frames, gop_data = gop_reader.get_gop(i)
      dec.write(gop_data)
      i += num_frames
      read_work.append([skip_frames, num_frames])

      while len(read_work) >= IN_FLIGHT_GOPS:
        for v in readthing(): yield v

    dec.eos()

    while read_work:
      for v in readthing(): yield v


def FrameIterator(fn, pix_fmt, **kwargs):
  fr = FrameReader(fn, **kwargs)
  if isinstance(fr, GOPReader):
    for v in GOPFrameIterator(fr, pix_fmt, kwargs.get("multithreaded", True)): yield v
  else:
    for i in range(fr.frame_count):
      yield fr.get(i, pix_fmt=pix_fmt)[0]


def FrameWriter(ofn, frames, vid_fmt=FrameType.ffvhuff_mkv, pix_fmt="rgb24", framerate=20, multithreaded=False):
  if pix_fmt not in ("rgb24", "yuv420p"):
    raise NotImplementedError

  if vid_fmt == FrameType.ffv1_mkv:
    assert ofn.endswith(".mkv")
    vcodec = "ffv1"
  elif vid_fmt == FrameType.ffvhuff_mkv:
    assert ofn.endswith(".mkv")
    vcodec = "ffvhuff"
  else:
    raise NotImplementedError

  frame_gen = iter(frames)
  first_frame = next(frame_gen)

  # assert len(frames) > 1
  if pix_fmt == "rgb24":
    h, w = first_frame.shape[:2]
  elif pix_fmt == "yuv420p":
    w = first_frame.shape[1]
    h = 2*first_frame.shape[0]//3
  else:
    raise NotImplementedError

  compress_proc = subprocess.Popen(
    ["ffmpeg",
     "-threads", "0" if multithreaded else "1",
     "-y",
     "-framerate", str(framerate),
     "-vsync", "0",
     "-f", "rawvideo",
     "-pix_fmt", pix_fmt,
     "-s", "%dx%d" % (w, h),
     "-i", "pipe:0",
     "-threads", "0" if multithreaded else "1",
     "-f", "matroska",
     "-vcodec", vcodec,
     "-g", "0",
     ofn],
    stdin=subprocess.PIPE, stderr=open("/dev/null", "wb"))
  try:
    compress_proc.stdin.write(first_frame.tobytes())
    for frame in frame_gen:
      compress_proc.stdin.write(frame.tobytes())
    compress_proc.stdin.close()
  except:
    compress_proc.kill()
    raise

  assert compress_proc.wait() == 0

if __name__ == "__main__":
  fn = "cd:/1c79456b0c90f15a/2017-05-10--08-17-00/2/fcamera.hevc"
  f = FrameReader(fn)
  # print f.get(0, 1).shape
  # print f.get(15, 1).shape
  for v in GOPFrameIterator(f, "yuv420p"):
    print(v)
