#!/usr/bin/env python3

import sys
import argparse
import multiprocessing
import rerun as rr
import rerun.blueprint as rrb
from functools import partial
from collections import defaultdict

from cereal.services import SERVICE_LIST
from openpilot.tools.rerun.camera_reader import probe_packet_info, CameraReader, CameraConfig, CameraType
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentRange


NUM_CPUS = multiprocessing.cpu_count()
DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
RR_TIMELINE_NAME = "Timeline"
RR_WIN = "openpilot logs"


"""
Relevant upstream Rerun issues:
- loading videos directly: https://github.com/rerun-io/rerun/issues/6532
"""

class Rerunner:
  def __init__(self, route, segment_range, camera_config):
    self.lr = LogReader(route_or_segment_name)

    # hevc files don't have start_time. We get it from qcamera.ts
    start_time = 0
    dat = probe_packet_info(route.qcamera_paths()[0])
    for d in dat:
      if d.startswith("pts_time="):
        start_time = float(d.split('=')[1])
        break

    qcam, fcam, ecam, dcam = camera_config
    self.camera_readers = {}
    if qcam:
      self.camera_readers[CameraType.qcam] = CameraReader(route.qcamera_paths(), start_time, segment_range.seg_idxs)
    if fcam:
      self.camera_readers[CameraType.fcam] = CameraReader(route.camera_paths(), start_time, segment_range.seg_idxs)
    if ecam:
      self.camera_readers[CameraType.ecam] = CameraReader(route.ecamera_paths(), start_time, segment_range.seg_idxs)
    if dcam:
      self.camera_readers[CameraType.dcam] = CameraReader(route.dcamera_paths(), start_time, segment_range.seg_idxs)

  def _create_blueprint(self):
    blueprint = None
    service_views = []

    for topic in sorted(SERVICE_LIST.keys()):
      View = rrb.TimeSeriesView if topic != "thumbnail" else rrb.Spatial2DView
      service_views.append(View(name=topic, origin=f"/{topic}/", visible=False))
      rr.log(topic, rr.SeriesLine(name=topic), timeless=True)

    center_view = [rrb.Vertical(*service_views, name="streams")]
    if len(self.camera_readers):
      center_view.append(rrb.Vertical(*[rrb.Spatial2DView(name=cam_type, origin=cam_type) for cam_type in self.camera_readers.keys()], name="cameras"))

    blueprint = rrb.Blueprint(
      rrb.Horizontal(
        *center_view
      ),
      rrb.SelectionPanel(expanded=False),
      rrb.TimePanel(expanded=False),
    )
    return blueprint

  @staticmethod
  def _parse_msg(msg, parent_key=''):
    stack = [(msg, parent_key)]
    while stack:
      current_msg, current_parent_key = stack.pop()
      if isinstance(current_msg, list):
        for index, item in enumerate(current_msg):
          new_key = f"{current_parent_key}/{index}"
          if isinstance(item, (int, float)):
            yield new_key, item
          elif isinstance(item, dict):
            stack.append((item, new_key))
      elif isinstance(current_msg, dict):
        for key, value in current_msg.items():
          new_key = f"{current_parent_key}/{key}"
          if isinstance(value, (int, float)):
            yield new_key, value
          elif isinstance(value, dict):
            stack.append((value, new_key))
          elif isinstance(value, list):
            for index, item in enumerate(value):
              if isinstance(item, (int, float)):
                yield f"{new_key}/{index}", item
      else:
        pass  # Not a plottable value

  @staticmethod
  @rr.shutdown_at_exit
  def _process_log_msgs(blueprint, lr):
    rr.init(RR_WIN)
    rr.connect()
    rr.send_blueprint(blueprint)

    log_msgs = defaultdict(lambda: defaultdict(list))
    for msg in lr:
      msg_type = msg.which()

      if msg_type == "thumbnail":
        continue

      for entity_path, dat in Rerunner._parse_msg(msg.to_dict()[msg_type], msg_type):
        log_msgs[entity_path]["times"].append(msg.logMonoTime)
        log_msgs[entity_path]["data"].append(dat)

    for entity_path, log_msg in log_msgs.items():
      rr.send_columns(
        entity_path,
        times=[rr.TimeNanosColumn(RR_TIMELINE_NAME, log_msg["times"])],
        components=[rr.components.ScalarBatch(log_msg["data"])]
      )

    return []

  @staticmethod
  @rr.shutdown_at_exit
  def _process_cam_readers(blueprint, cam_type, h, w, fr):
    rr.init(RR_WIN)
    rr.connect()
    rr.send_blueprint(blueprint)

    for ts, frame in fr:
      rr.set_time_nanos(RR_TIMELINE_NAME, int(ts * 1e9))
      rr.log(cam_type, rr.Image(bytes=frame, width=w, height=h, pixel_format=rr.PixelFormat.NV12))

  def load_data(self):
    rr.init(RR_WIN, spawn=True)

    startup_blueprint = self._create_blueprint()
    self.lr.run_across_segments(NUM_CPUS, partial(self._process_log_msgs, startup_blueprint), desc="Log messages")
    for cam_type, cr in self.camera_readers.items():
      cr.run_across_segments(NUM_CPUS, partial(self._process_cam_readers, startup_blueprint, cam_type, cr.h, cr.w), desc=cam_type)

    rr.send_blueprint(self._create_blueprint())


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="A helper to run rerun on openpilot routes",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--qcam", action="store_true", help="Show low-res road camera")
  parser.add_argument("--fcam", action="store_true", help="Show driving camera")
  parser.add_argument("--ecam", action="store_true", help="Show wide camera")
  parser.add_argument("--dcam", action="store_true", help="Show driver monitoring camera")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot")
  args = parser.parse_args()

  if not args.demo and not args.route_or_segment_name:
    parser.print_help()
    sys.exit()

  camera_config = CameraConfig(args.qcam, args.fcam, args.ecam, args.dcam)
  route_or_segment_name = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()

  sr = SegmentRange(route_or_segment_name)
  r = Route(sr.route_name)

  hevc_requested = any(camera_config[1:])
  if len(sr.seg_idxs) > 1 and hevc_requested:
    print("You're requesting more than 1 segment with hevc videos, " + \
          "please be aware that might take a lot of memory " + \
          "since rerun isn't yet well supported for high resolution video logging")
    response = input("Do you wish to continue? (Y/n): ")
    if response.strip().lower() != "y":
      sys.exit()

  rerunner = Rerunner(r, sr, camera_config)
  rerunner.load_data()

