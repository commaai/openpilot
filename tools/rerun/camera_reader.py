import tqdm
import subprocess
import multiprocessing
from enum import StrEnum
from functools import partial
from collections import namedtuple

from openpilot.tools.lib.framereader import ffprobe

CameraConfig = namedtuple("CameraConfig", ["qcam", "fcam", "ecam", "dcam"])

class CameraType(StrEnum):
  qcam = "qcamera"
  fcam = "fcamera"
  ecam = "ecamera"
  dcam = "dcamera"


def probe_packet_info(camera_path):
  args = ["ffprobe", "-v", "quiet", "-show_packets", "-probesize", "10M", camera_path]
  dat = subprocess.check_output(args)
  dat = dat.decode().split()
  return dat


class _FrameReader:
  def __init__(self, camera_path, segment, h, w, start_time):
    self.camera_path = camera_path
    self.segment = segment
    self.h = h
    self.w = w
    self.start_time = start_time

    self.ts = self._get_ts()

  def _read_stream_nv12(self):
    frame_sz = self.w * self.h * 3 // 2
    proc = subprocess.Popen(
             ["ffmpeg", "-v", "quiet", "-i", self.camera_path, "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
             stdin=subprocess.PIPE,
             stdout=subprocess.PIPE,
             stderr=subprocess.DEVNULL
           )
    try:
      while True:
        dat = proc.stdout.read(frame_sz)
        if len(dat) == 0:
          break
        yield dat
    finally:
      proc.kill()

  def _get_ts(self):
    dat = probe_packet_info(self.camera_path)
    try:
      ret = [float(d.split('=')[1]) for d in dat if d.startswith("pts_time=")]
    except ValueError:
      # pts_times aren't available. Infer timestamps from duration_times
      ret = [d for d in dat if d.startswith("duration_time")]
      ret = [float(d.split('=')[1])*(i+1)+(self.segment*60)+self.start_time for i, d in enumerate(ret)]
    return ret

  def __iter__(self):
    for i, frame in enumerate(self._read_stream_nv12()):
      yield self.ts[i], frame


class CameraReader:
  def __init__(self, camera_paths, start_time, seg_idxs):
    self.seg_idxs = seg_idxs
    self.camera_paths = camera_paths
    self.start_time = start_time

    probe = ffprobe(camera_paths[0])["streams"][0]
    self.h = probe["height"]
    self.w = probe["width"]

    self.__frs = {}

  def _get_fr(self, i):
    if i not in self.__frs:
      self.__frs[i] = _FrameReader(self.camera_paths[i], segment=i, h=self.h, w=self.w, start_time=self.start_time)
    return self.__frs[i]

  def _run_on_segment(self, func, i):
    return func(self._get_fr(i))

  def run_across_segments(self, num_processes, func, desc=None):
    with multiprocessing.Pool(num_processes) as pool:
      num_segs = len(self.seg_idxs)
      for _ in tqdm.tqdm(pool.imap_unordered(partial(self._run_on_segment, func), self.seg_idxs), total=num_segs, desc=desc):
        continue

