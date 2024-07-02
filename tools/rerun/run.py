#!/usr/bin/env python3
import sys
import argparse
import tqdm
import multiprocessing
import rerun as rr
import rerun.blueprint as rrb
from functools import partial
from collections import defaultdict

from openpilot.tools.lib.logreader import LogReader
from cereal.services import SERVICE_LIST


NUM_CPUS = multiprocessing.cpu_count()
DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"


class Rerunner:
  def __init__(self, feature_rich, route_or_segment_name):
    self.feature_rich = feature_rich
    self.lr = LogReader(route_or_segment_name)

  @staticmethod
  def parse_log(msg, parent_key=""):
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
  def log_thumbnail(thumbnailMsg):
    bytesImgData = thumbnailMsg.get("thumbnail")
    rr.log("/thumbnail", rr.ImageEncoded(contents=bytesImgData))

  def start_rerun(self):
    self.createBlueprint()
    rr.init("rerun_test", spawn=True)

  def createBlueprint(self):
    blueprint = None
    service_views = []
    View = rrb.TimeSeriesView if self.feature_rich else rrb.Spatial2DView

    for topic in sorted(SERVICE_LIST.keys()):
      if topic == "thumbnail":
        continue
      if feature_rich:
        rr.log(topic, rr.SeriesLine(name=topic), static=True)
      service_views.append(View(name=topic, origin=f"/{topic}", visible=False))

    blueprint = rrb.Blueprint(
      rrb.Vertical(*service_views),
      rrb.Spatial2DView(name="thumbnail", origin="/thumbnail", visible=False),
      rrb.SelectionPanel(expanded=False),
      rrb.TimePanel(expanded=False),
    )
    self.blueprint = blueprint

  @staticmethod
  @rr.shutdown_at_exit
  def process(blueprint, lr):
    rr.init("rerun_test")
    rr.connect(default_blueprint=blueprint)
    log_data = defaultdict(list)

    for msg in lr:
      time = msg.logMonoTime * 1e-9

      if msg.which() == "thumbnail":
        Rerunner.log_thumbnail(msg.to_dict()[msg.which()])
        continue

      for path, data_pt in Rerunner.parse_log(msg.to_dict()[msg.which()], msg.which()):
        if feature_rich:
          rr.log(path, rr.Scalar(data_pt))
        else:
          log_data[path].append([time, -data_pt]) # negative since it follows screen coordinate

    return [log_data]

  def log_msgs(self):
    self.start_rerun()
    msg_from_segments = self.lr.run_across_segments(NUM_CPUS, partial(self.process, self.blueprint))

    if not self.feature_rich:
      log_data = defaultdict(list)
      for batch in msg_from_segments:
        for k, v in batch.items():
          log_data[k].extend(v)

      print("Parsing Log...")
      for path, data_pts in tqdm.tqdm(log_data.items()):
        rr.log(path, rr.LineStrips2D(data_pts, radii=0.005))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper to run rerun on openpilot routes",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  # TODO: remove this mode once upstream is fixed. Issue tracker: https://github.com/rerun-io/rerun/issues/5967
  parser.add_argument("--feature_rich", action="store_true", help="Use this option for the full, but expensive, functionalities of Rerun")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  args = parser.parse_args()
  feature_rich = args.feature_rich
  route_or_segment_name = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()

  rerunner = Rerunner(feature_rich, route_or_segment_name)
  rerunner.log_msgs()

