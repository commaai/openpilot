#!/usr/bin/env python3
import sys
from collections import Counter
from pprint import pprint
from tqdm import tqdm

from tools.lib.route import Route
from tools.lib.logreader import LogReader

if __name__ == "__main__":
  r = Route(sys.argv[1])

  cnt_valid: Counter = Counter()
  cnt_events: Counter = Counter()

  for q in tqdm(r.qlog_paths()):
    lr = list(LogReader(q))
    for msg in lr:
      if msg.which() == 'carEvents':
        for e in msg.carEvents:
          cnt_events[e.name] += 1
      if not msg.valid:
        cnt_valid[msg.which()] += 1

  print("Events")
  pprint(cnt_events)

  print("\n\n")
  print("Not valid")
  pprint(cnt_valid)
