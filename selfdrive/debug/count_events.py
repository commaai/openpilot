#!/usr/bin/env python3
import sys
from collections import Counter
from pprint import pprint
from tqdm import tqdm

from tools.lib.route import Route
from tools.lib.logreader import LogReader

if __name__ == "__main__":
  r = Route(sys.argv[1])

  cnt: Counter = Counter()
  for q in tqdm(r.qlog_paths()):
    lr = LogReader(q)
    car_events = [m for m in lr if m.which() == 'carEvents']
    for car_event in car_events:
      for e in car_event.carEvents:
        cnt[e.name] += 1
  pprint(cnt)
