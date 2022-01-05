#!/usr/bin/env python3
import os
import zmq
import json
import time
from pathlib import Path
from collections import deque
from datetime import datetime
from selfdrive.hardware import PC
from selfdrive.swaglog import cloudlog
from cereal.messaging import SubMaster
from common.file_helpers import atomic_write_on_fs_tmp

FLUSH_TIME_S = 60
STATS_SOCKET = "ipc:///tmp/stats"
if PC:
  STATS_DIR = os.path.join(str(Path.home()), ".comma", "stats")
else:
  STATS_DIR = "/data/stats/"

class METRIC_TYPE:
  GAUGE = 'g'

class StatLog:
  def __init__(self):
    self.pid = None

  def connect(self):
    self.zctx = zmq.Context()
    self.sock = self.zctx.socket(zmq.PUSH)
    self.sock.setsockopt(zmq.LINGER, 10)
    self.sock.connect(STATS_SOCKET)
    self.pid = os.getpid()

  def gauge(self, name: str, value: float):
    if os.getpid() != self.pid:
      self.connect()
    self.sock.send_string(f"{name}:{value}|{METRIC_TYPE.GAUGE}", zmq.NOBLOCK)


def run_daemon():
  # open statistics socket
  ctx = zmq.Context().instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind(STATS_SOCKET)

  # initialize stats directory
  Path(STATS_DIR).mkdir(parents=True, exist_ok=True)

  # subscribe to deviceState for started state
  sm = SubMaster(['deviceState'])

  last_flush_time = time.monotonic()
  gauges = {}
  started_prev = False
  while True:
    try:
      metric = sock.recv_string(zmq.NOBLOCK)
      try:
        metric_type = metric.split('|')[1]
        metric_name = metric.split(':')[0]
        metric_value = metric.split('|')[0].split(':')[1]

        if metric_type == METRIC_TYPE.GAUGE:
          gauges[metric_name] = metric_value
        else:
          cloudlog.error(f"unknown metric type: {metric_type}")
      except Exception:
        cloudlog.error(f"malformed metric: {metric}")
    except zmq.error.Again:
      time.sleep(1e-3)

    started = sm['deviceState'].started
    # flush when started state changes or after FLUSH_TIME_S
    if (time.monotonic() > last_flush_time + FLUSH_TIME_S) or (started != started_prev):
      flush_result = {
        'time_utc': int(datetime.utcnow().timestamp()),
        'started': started,
        'gauges': {**gauges},
      }

      # clear intermediate data
      gauges = {}

      stats_path = os.path.join(STATS_DIR, str(flush_result['time_utc']))
      with atomic_write_on_fs_tmp(stats_path) as f:
        f.write(json.dumps(flush_result))

      last_flush_time = time.monotonic()
    started_prev = started

if __name__ == "__main__":
  run_daemon()
else:
  statlog = StatLog()
