#!/usr/bin/env python3
import sys
import argparse
import multiprocessing
import rerun as rr
import rerun.blueprint as rrb
from functools import partial
import subprocess
import numpy as np
from enum import IntEnum, StrEnum
from collections import namedtuple

from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route
from openpilot.tools.lib.framereader import FrameIterator, ffprobe
from cereal.services import SERVICE_LIST


NUM_CPUS = multiprocessing.cpu_count()
DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"

"""
TODO:
  [x] design
  [x] one probe
  [x] hevc vids
  [x] better naming
  [x] size hint
  [] multiprocessing qcameras speed for speed
  [] type hint
"""

def log_msg(msg, parent_key=''):
  stack = [(msg, parent_key)]
  while stack:
    current_msg, current_parent_key = stack.pop()
    if isinstance(current_msg, list):
      for index, item in enumerate(current_msg):
        new_key = f"{current_parent_key}/{index}"
        if isinstance(item, (int, float)):
          rr.log(str(new_key), rr.Scalar(item))
        elif isinstance(item, dict):
          stack.append((item, new_key))
    elif isinstance(current_msg, dict):
      for key, value in current_msg.items():
        new_key = f"{current_parent_key}/{key}"
        if isinstance(value, (int, float)):
          rr.log(str(new_key), rr.Scalar(value))
        elif isinstance(value, dict):
          stack.append((value, new_key))
        elif isinstance(value, list):
          for index, item in enumerate(value):
            if isinstance(item, (int, float)):
              rr.log(f"{new_key}/{index}", rr.Scalar(item))
    else:
      pass  # Not a plottable value


class FrameType(IntEnum):
  h264_stream = 1
  h265_stream = 2


class FrameReaderWithTimestamps:
  def __init__(self, camera_path, segment, h, w, frame_type):
    self.camera_path = camera_path
    self.segment = segment
    self.h = h
    self.w = w
    self.frame_type = frame_type

    if frame_type == FrameType.h265_stream:
      self.__frame_iter = FrameIterator(self.camera_path, "nv12")
    elif frame_type == FrameType.h264_stream:
      self.__frame_iter = self.read_h264_stream()
    else:
      raise NotImplementedError(frame_type)

    self.ts = self._get_ts()


  def _get_ts(self):
    args = ["ffprobe", "-show_packets", "-probesize", "10M", self.camera_path]
    dat = subprocess.check_output(args)
    dat = dat.decode().split()
    try:
      ret = [float(d.split('=')[1]) for d in dat if d.startswith("pts_time=")]
    except ValueError:
      # pts_times aren't available. Infer timestamps from duration_times
      ret = [d for d in dat if d.startswith("duration_time")]
      ret = [float(d.split('=')[1])*(i+1)+(self.segment*60) for i, d in enumerate(ret)]
    return ret


  def read_h264_stream(self):
    frame_sz = self.w * self.h * 3 // 2

    proc = subprocess.Popen(
      ['ffmpeg', '-v', 'quiet', '-i', self.camera_path, '-f', 'rawvideo', '-pix_fmt', 'nv12', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    while True:
      dat = proc.stdout.read(frame_sz)
      if len(dat) == 0:
        break
      yield dat


  def __iter__(self):
    for i, frame in enumerate(self.__frame_iter):
      if type(frame) == np.ndarray:
        frame = frame.data.tobytes()
      yield self.ts[i], frame


class CameraReader:
  def __init__(self, camera_paths, frame_type=FrameType.h265_stream):
    self.camera_paths = camera_paths
    self.frame_type = frame_type

    probe = ffprobe(camera_paths[0])["streams"][0]
    self.h = probe["height"]
    self.w = probe["width"]

    self.__frs = {}


  def _get_fr(self, i):
    if i not in self.__frs:
      self.__frs[i] = FrameReaderWithTimestamps(self.camera_paths[i], segment=i, h=self.h, w=self.w, frame_type=self.frame_type)
    return self.__frs[i]


  def __iter__(self):
    for i in range(len(self.camera_paths)):
      yield from self._get_fr(i)


CameraConfig = namedtuple("CameraConfig", ["qcam", "fcam", "ecam", "dcam"])


class CameraType(StrEnum):
  qcam = "qcamera"
  fcam = "fcamera"
  ecam = "ecamera"
  dcam = "dcamera"


class Rerunner:
  def __init__(self, route_or_segment_name, camera_config, enabled_services):
    self.enabled_services = [s.lower() for s in enabled_services]
    self.lr = LogReader(route_or_segment_name)
    self.r = Route(route_or_segment_name)

    self.qcam, self.fcam, self.ecam, self.dcam = camera_config

    self.camera_readers = {}
    if self.qcam:
      self.camera_readers[CameraType.qcam] = CameraReader(self.r.qcamera_paths(), FrameType.h264_stream)
    if self.fcam:
      self.camera_readers[CameraType.fcam] = CameraReader(self.r.camera_paths())
    if self.ecam:
      self.camera_readers[CameraType.ecam] = CameraReader(self.r.ecamera_paths())
    if self.dcam:
      self.camera_readers[CameraType.dcam] = CameraReader(self.r.dcamera_paths())


  def _start_rerun(self):
    self.blueprint = self._create_blueprint()
    rr.init("rerun_test", spawn=True)


  def _create_blueprint(self):
    blueprint = None
    service_views = []

    for topic in sorted(SERVICE_LIST.keys()):
      if topic.lower() not in self.enabled_services:
        continue
      View = rrb.TimeSeriesView if topic == "thumbnail" else rrb.Spatial2DView
      service_views.append(View(name=topic, origin=f"/{topic}/", visible=False))
      rr.log(topic, rr.SeriesLine(name=topic), timeless=True)

    blueprint = rrb.Blueprint(
      *service_views,
      *[rrb.Spatial2DView(name=cam_type, origin=cam_type, visible=False) for cam_type in self.camera_readers.keys()],
      rrb.SelectionPanel(expanded=False),
      rrb.TimePanel(expanded=False)
    )
    return blueprint


  @staticmethod
  @rr.shutdown_at_exit
  def _process_log_msgs(blueprint, enabled_services, lr):
    rr.init("rerun_test")
    rr.connect(default_blueprint=blueprint)

    ret = []
    for msg in lr:
      ret.append(msg)
      rr.set_time_nanos("TIMELINE", msg.logMonoTime)
      msg_type = msg.which()

      if msg_type.lower() not in enabled_services:
        continue

      if msg_type != "thumbnail":
        log_msg(msg.to_dict()[msg.which()], msg.which())
      else:
        rr.log("/thumbnail", rr.ImageEncoded(contents=msg.to_dict()[msg.which()].get("thumbnail")))

    return ret


  def load_data(self):
    self._start_rerun()
    self.lr.run_across_segments(NUM_CPUS, partial(self._process_log_msgs, self.blueprint, self.enabled_services))
    for cam_type, cr in self.camera_readers.items():
      size_hint = (cr.h, cr.w)
      for ts, frame in cr:
        rr.set_time_nanos("TIMELINE", int(ts * 1e9))
        rr.log(cam_type, rr.ImageEncoded(contents=frame,format=rr.ImageFormat.NV12(size_hint)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="A helper to run rerun on openpilot routes",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--qcam", action="store_true", help="Log decimated driving camera")
  parser.add_argument("--fcam", action="store_true", help="Log driving camera")
  parser.add_argument("--ecam", action="store_true", help="Log wide camera")
  parser.add_argument("--dcam", action="store_true", help="Log driver monitoring camera")
  parser.add_argument("--services", default=[], nargs='*', help="Specify openpilot services that will be logged. No services will be logged if not specified")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  args = parser.parse_args()
  route_or_segment_name = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  camera_config = CameraConfig(args.qcam, args.fcam, args.ecam, args.dcam)

  rerunner = Rerunner(route_or_segment_name, camera_config, args.services)
  rerunner.load_data()

