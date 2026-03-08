#!/usr/bin/env python3
import sys
import time
import numpy as np
import datetime
from collections.abc import MutableSequence
from collections import defaultdict

import cereal.messaging as messaging


if __name__ == "__main__":
  ts: defaultdict[str, MutableSequence[float]] = defaultdict(list)
  socks = {s: messaging.sub_sock(s, conflate=False) for s in sys.argv[1:]}
  try:
    st = time.monotonic()
    while True:
      print()
      for s, sock in socks.items():
        msgs = messaging.drain_sock(sock)
        for m in msgs:
          ts[s].append(m.logMonoTime / 1e6)

        if len(ts[s]) > 2:
          d = np.diff(ts[s])[-100:]
          print(f"{s:25} {np.mean(d):7.2f} {np.std(d):7.2f} {np.max(d):7.2f} {np.min(d):7.2f}")
      time.sleep(1)
  except KeyboardInterrupt:
    print("\n")
    print("="*5, "timing summary", "="*5)
    for s, sock in socks.items():
      msgs = messaging.drain_sock(sock)
      if len(ts[s]) > 2:
        d = np.diff(ts[s])
        print(f"{s:25} {np.mean(d):7.2f} {np.std(d):7.2f} {np.max(d):7.2f} {np.min(d):7.2f}")
    print("="*5, datetime.timedelta(seconds=time.monotonic()-st), "="*5)
