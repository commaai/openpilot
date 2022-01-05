#!/usr/bin/env python3
import os
import zmq
import json
import time
from pathlib import Path
from collections import deque
from datetime import datetime
from selfdrive.hardware import PC
from cereal.messaging import SubMaster
from common.file_helpers import atomic_write_on_fs_tmp


AGGREGATION_TIME_S = 60
STATS_SOCKET = "ipc:///tmp/stats"
if PC:
  STATS_DIR = os.path.join(str(Path.home()), ".comma", "stats")
else:
  STATS_DIR = "/data/stats/"


class StatLog:
  def __init__(self):
    self.pid = None

  def connect(self):
    self.zctx = zmq.Context()
    self.sock = self.zctx.socket(zmq.PUSH)
    self.sock.setsockopt(zmq.LINGER, 10)
    self.sock.connect(STATS_SOCKET)
    self.pid = os.getpid()

  def log(self, name: str, value: float):
    if os.getpid() != self.pid:
      self.connect()
    self.sock.send_string(json.dumps({"name": name, "val": value}), zmq.NOBLOCK)


def run_daemon():
  def aggregate(values: deque):
    samples = list(values)
    samples.sort()

    _len = len(samples)
    _sum = sum(samples)

    aggregates =  {
      'count': _len,
      'max': samples[-1],
      'min': samples[0],
      'mean': _sum / _len,
    }

    for percentile in [0.05, 0.25, 0.5, 0.75, 0.95]:
      aggregates[f"p{int(percentile * 100)}"] = samples[int(round(percentile * _len - 1))]

    return aggregates

  # open statistics socket
  ctx = zmq.Context().instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind(STATS_SOCKET)

  # initialize stats directory
  Path(STATS_DIR).mkdir(parents=True, exist_ok=True)

  # subscribe to deviceState for started state
  sm = SubMaster(['deviceState'])

  last_aggregate_time = time.monotonic()
  values = {}
  started_prev = False
  while True:
    try:
      dat = json.loads(b''.join(sock.recv_multipart(zmq.NOBLOCK)))
      if dat['name'] not in values.keys():
        values[dat['name']] = deque()
      values[dat['name']].append(dat['val'])
    except zmq.error.Again:
      time.sleep(1e-3)

    started = sm['deviceState'].started
    # aggregate when started state changes or after AGGREGATION_TIME_S
    if (time.monotonic() > last_aggregate_time + AGGREGATION_TIME_S) or (started != started_prev):
      aggregates = {
        'time_utc': int(datetime.utcnow().timestamp()),
        'started': started,
      }

      for key in values.keys():
        if len(values[key]) > 0:
          aggregates[key] = aggregate(values[key])
        values[key].clear()

      stats_path = os.path.join(STATS_DIR, str(aggregates['time_utc']))
      with atomic_write_on_fs_tmp(stats_path) as f:
        f.write(json.dumps(aggregates))

      last_aggregate_time = time.monotonic()
    started_prev = started

if __name__ == "__main__":
  run_daemon()
else:
  statlog = StatLog()
