#!/usr/bin/env python3
import argparse
import numpy as np
import time
from collections import defaultdict, deque
from collections.abc import MutableSequence

import cereal.messaging as messaging


if __name__ == "__main__":
  context = messaging.Context()
  poller = messaging.Poller()

  parser = argparse.ArgumentParser()
  parser.add_argument("socket", type=str, nargs='*', help="socket name")
  args = parser.parse_args()

  socket_names = args.socket
  sockets = {}

  rcv_times: defaultdict[str, MutableSequence[float]] = defaultdict(lambda: deque(maxlen=100))
  valids: defaultdict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=100))

  t = time.monotonic()
  for name in socket_names:
    sock = messaging.sub_sock(name, poller=poller)
    sockets[sock] = name

  prev_print = t
  while True:
    for socket in poller.poll(100):
      msg = messaging.recv_one(socket)
      if msg is None:
        continue

      name = msg.which()

      t = time.monotonic()
      rcv_times[name].append(msg.logMonoTime / 1e9)
      valids[name].append(msg.valid)

    if t - prev_print > 1:
      print()
      for name in socket_names:
        dts = np.diff(rcv_times[name])
        mean = np.mean(dts)
        print(f"{name}: Freq {1.0 / mean:.2f} Hz, Min {np.min(dts) / mean * 100:.2f}%, Max {np.max(dts) / mean * 100:.2f}%, valid ", all(valids[name]))

      prev_print = t
