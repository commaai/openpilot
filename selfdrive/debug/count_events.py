#!/usr/bin/env python3
import sys
from collections import Counter
from pprint import pprint
from tqdm import tqdm

from cereal.services import service_list
from tools.lib.route import Route
from tools.lib.logreader import LogReader

if __name__ == "__main__":
  r = Route(sys.argv[1])

  cnt_valid: Counter = Counter()
  cnt_events: Counter = Counter()

  cams = [s for s in service_list if s.endswith('CameraState')]
  cnt_cameras = dict.fromkeys(cams, 0)

  for q in tqdm(r.qlog_paths()):
    if q is None:
      continue
    lr = list(LogReader(q))
    for msg in lr:
      if msg.which() == 'carEvents':
        for e in msg.carEvents:
          cnt_events[e.name] += 1
      elif msg.which() in cams:
        cnt_cameras[msg.which()] += 1

      if not msg.valid:
        cnt_valid[msg.which()] += 1


  print("Events")
  pprint(cnt_events)

  print("\n")
  print("Not valid")
  pprint(cnt_valid)

  print("\n")
  print("Cameras")
  for k, v in cnt_cameras.items():
    print("  ", k.ljust(20), v)
