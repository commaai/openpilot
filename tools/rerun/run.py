#!/usr/bin/env python3
import sys
import argparse
import multiprocessing
import tqdm
import subprocess
import rerun as rr
import rerun.blueprint as rrb
from functools import partial
from enum import StrEnum
from collections import namedtuple

from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentRange
from openpilot.tools.lib.framereader import ffprobe
from cereal.services import SERVICE_LIST


NUM_CPUS = multiprocessing.cpu_count()
DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
RR_TIMELINE_NAME = "Timeline"
RR_WIN = "rerun_test"


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
    self.camera_paths = [camera_paths[i] for i in seg_idxs]
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

  def run_across_segments(self, num_processes, func):
    with multiprocessing.Pool(num_processes) as pool:
      num_segs = len(self.camera_paths)
      for _ in tqdm.tqdm(pool.imap_unordered(partial(self._run_on_segment, func), range(num_segs)), total=num_segs):
        continue


CameraConfig = namedtuple("CameraConfig", ["qcam", "fcam", "ecam", "dcam"])


class CameraType(StrEnum):
  qcam = "qcamera"
  fcam = "fcamera"
  ecam = "ecamera"
  dcam = "dcamera"


class Rerunner:
  def __init__(self, route_or_segment_name, camera_config, enabled_services):
    self.enabled_services = [s.lower() for s in enabled_services]
    self.log_all = "all" in self.enabled_services
    self.lr = LogReader(route_or_segment_name)

    sr = SegmentRange(route_or_segment_name)
    r = Route(sr.route_name)

    self.qcam, self.fcam, self.ecam, self.dcam = camera_config

    # hevc files don't have start_time. We get it from qcamera.ts
    start_time = 0
    dat = probe_packet_info(r.qcamera_paths()[0])
    for d in dat:
      if d.startswith("pts_time="):
        start_time = float(d.split('=')[1])
        break

    self.camera_readers = {}
    if self.qcam:
      self.camera_readers[CameraType.qcam] = CameraReader(r.qcamera_paths(), start_time, sr.seg_idxs)
    if self.fcam:
      self.camera_readers[CameraType.fcam] = CameraReader(r.camera_paths(), start_time, sr.seg_idxs)
    if self.ecam:
      self.camera_readers[CameraType.ecam] = CameraReader(r.ecamera_paths(), start_time, sr.seg_idxs)
    if self.dcam:
      self.camera_readers[CameraType.dcam] = CameraReader(r.dcamera_paths(), start_time, sr.seg_idxs)

  def _start_rerun(self):
    self.blueprint = self._create_blueprint()
    rr.init(RR_WIN, spawn=True)

  def _create_blueprint(self):
    blueprint = None
    service_views = []

    log_msg_visible = len(self.enabled_services) <= 3 and not self.log_all
    for topic in sorted(SERVICE_LIST.keys()):
      if not self.log_all and topic.lower() not in self.enabled_services:
        continue
      View = rrb.TimeSeriesView if topic != "thumbnail" else rrb.Spatial2DView
      service_views.append(View(name=topic, origin=f"/{topic}/", visible=log_msg_visible))
      rr.log(topic, rr.SeriesLine(name=topic), timeless=True)

    blueprint = rrb.Blueprint(
      rrb.Horizontal(
        rrb.Vertical(*service_views),
        rrb.Vertical(*[rrb.Spatial2DView(name=cam_type, origin=cam_type) for cam_type in self.camera_readers.keys()]),
      ),
      rrb.SelectionPanel(expanded=False),
      rrb.TimePanel(expanded=False)
    )
    return blueprint

  @staticmethod
  def _log_msg(msg, parent_key=''):
    stack = [(msg, parent_key)]
    while stack:
      current_msg, current_parent_key = stack.pop()
      if isinstance(current_msg, list):
        for index, item in enumerate(current_msg):
          new_key = f"{current_parent_key}/{index}"
          if isinstance(item, (int, float)):
            rr.log(new_key, rr.Scalar(item))
          elif isinstance(item, dict):
            stack.append((item, new_key))
      elif isinstance(current_msg, dict):
        for key, value in current_msg.items():
          new_key = f"{current_parent_key}/{key}"
          if isinstance(value, (int, float)):
            rr.log(new_key, rr.Scalar(value))
          elif isinstance(value, dict):
            stack.append((value, new_key))
          elif isinstance(value, list):
            for index, item in enumerate(value):
              if isinstance(item, (int, float)):
                rr.log(f"{new_key}/{index}", rr.Scalar(item))
      else:
        pass  # Not a plottable value

  @staticmethod
  @rr.shutdown_at_exit
  def _process_log_msgs(blueprint, enabled_services, log_all, lr):
    rr.init(RR_WIN)
    rr.connect(default_blueprint=blueprint)

    for msg in lr:
      rr.set_time_nanos(RR_TIMELINE_NAME, msg.logMonoTime)
      msg_type = msg.which()

      if not log_all and msg_type.lower() not in enabled_services:
        continue

      if msg_type != "thumbnail":
        Rerunner._log_msg(msg.to_dict()[msg.which()], msg.which())
      else:
        rr.log("/thumbnail", rr.ImageEncoded(contents=msg.to_dict()[msg.which()].get("thumbnail")))

    return []

  @staticmethod
  @rr.shutdown_at_exit
  def _process_cam_readers(blueprint, cam_type, h, w, fr):
    rr.init(RR_WIN)
    rr.connect(default_blueprint=blueprint)

    size_hint = (h, w)
    for ts, frame in fr:
      rr.set_time_nanos(RR_TIMELINE_NAME, int(ts * 1e9))
      rr.log(cam_type, rr.ImageEncoded(contents=frame,format=rr.ImageFormat.NV12(size_hint)))

  def load_data(self):
    self._start_rerun()
    if len(self.enabled_services) > 0:
      self.lr.run_across_segments(NUM_CPUS, partial(self._process_log_msgs, self.blueprint, self.enabled_services, self.log_all))
    for cam_type, cr in self.camera_readers.items():
      cr.run_across_segments(NUM_CPUS, partial(self._process_cam_readers, self.blueprint, cam_type, cr.h, cr.w))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="A helper to run rerun on openpilot routes",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--qcam", action="store_true", help="Log decimated driving camera")
  parser.add_argument("--fcam", action="store_true", help="Log driving camera")
  parser.add_argument("--ecam", action="store_true", help="Log wide camera")
  parser.add_argument("--dcam", action="store_true", help="Log driver monitoring camera")
  parser.add_argument("--print_services", action="store_true", help="List out openpilot services")
  parser.add_argument("--services", default=[], nargs='*', help="Specify openpilot services that will be logged.\
                                                                No service will be logged if not specified.\
                                                                To log all services include 'all' as one of your services")
  parser.add_argument("--route", nargs='?', help="The route or segment name to plot")
  args = parser.parse_args()

  if not args.demo and not args.route:
    parser.print_help()
    sys.exit()

  if args.print_services:
    print("\n".join(SERVICE_LIST.keys()))
    sys.exit()

  route_or_segment_name = DEMO_ROUTE if args.demo else args.route.strip()
  camera_config = CameraConfig(args.qcam, args.fcam, args.ecam, args.dcam)

  rerunner = Rerunner(route_or_segment_name, camera_config, args.services)
  rerunner.load_data()

