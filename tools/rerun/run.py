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

def log_msg(msg, parent_key=''):
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

def createBlueprint():
  blueprint = None
  service_views = []
  for topic in sorted(SERVICE_LIST.keys()):
    if topic == "thumbnail":
      continue
    service_views.append(rrb.Spatial2DView(name=topic, origin=f"/{topic}/", visible=False))

  blueprint = rrb.Blueprint(
    rrb.Vertical(*service_views),
    rrb.Spatial2DView(name="thumbnail", origin="/thumbnail", visible=False),
    rrb.SelectionPanel(expanded=False),
    rrb.TimePanel(expanded=False),
  )
  return blueprint

def log_thumbnail(thumbnailMsg):
  bytesImgData = thumbnailMsg.get('thumbnail')
  rr.log("/thumbnail", rr.ImageEncoded(contents=bytesImgData))

@rr.shutdown_at_exit
def process(blueprint, lr):
  rr.init("rerun_test")
  rr.connect(default_blueprint=blueprint)

  log_data = defaultdict(list)

  for msg in lr:
    time = msg.logMonoTime * 1e-9
    if msg.which() != "thumbnail":
      for k, v in log_msg(msg.to_dict()[msg.which()], msg.which()):
          log_data[k].append([time, -v])
    else:
      log_thumbnail(msg.to_dict()[msg.which()])

  return [log_data]

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="A helper to run rerun on openpilot routes",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  args = parser.parse_args()

  blueprint = createBlueprint()
  rr.init("rerun_test", spawn=True, default_blueprint=blueprint)

  route_or_segment_name = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  print("Getting route log paths")
  lr = LogReader(route_or_segment_name)
  msg_from_segments = lr.run_across_segments(NUM_CPUS, partial(process, blueprint))

  log_data = defaultdict(list)
  for batch in msg_from_segments:
    for k, v in batch.items():
      log_data[k].extend(v)

  print("Parsing Log...")
  for path, data_points in tqdm.tqdm(log_data.items()):
    rr.log(path, rr.LineStrips2D(data_points, radii=0.005))

