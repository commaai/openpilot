#!/usr/bin/env python3
import os
import zmq
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import NoReturn

from common.params import Params
from cereal.messaging import SubMaster
from selfdrive.swaglog import cloudlog
from selfdrive.hardware import HARDWARE
from common.file_helpers import atomic_write_in_dir
from selfdrive.version import get_normalized_origin, get_short_branch, get_short_version, is_dirty
from selfdrive.loggerd.config import STATS_DIR, STATS_DIR_FILE_LIMIT, STATS_SOCKET, STATS_FLUSH_TIME_S


class METRIC_TYPE:
  GAUGE = 'g'

class StatLog:
  def __init__(self):
    self.pid = None

  def connect(self) -> None:
    self.zctx = zmq.Context()
    self.sock = self.zctx.socket(zmq.PUSH)
    self.sock.setsockopt(zmq.LINGER, 10)
    self.sock.connect(STATS_SOCKET)
    self.pid = os.getpid()

  def _send(self, metric: str) -> None:
    if os.getpid() != self.pid:
      self.connect()

    try:
      self.sock.send_string(metric, zmq.NOBLOCK)
    except zmq.error.Again:
      # drop :/
      pass

  def gauge(self, name: str, value: float) -> None:
    self._send(f"{name}:{value}|{METRIC_TYPE.GAUGE}")


def main() -> NoReturn:
  dongle_id = Params().get("DongleId", encoding='utf-8')
  def get_influxdb_line(measurement: str, value: float, timestamp: datetime, tags: dict) -> str:
    res = f"{measurement}"
    for k, v in tags.items():
      res += f",{k}={str(v)}"
    res += f" value={value},dongle_id=\"{dongle_id}\" {int(timestamp.timestamp() * 1e9)}\n"
    return res

  # open statistics socket
  ctx = zmq.Context().instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind(STATS_SOCKET)

  # initialize stats directory
  Path(STATS_DIR).mkdir(parents=True, exist_ok=True)

  # initialize tags
  tags = {
    'started': False,
    'version': get_short_version(),
    'branch': get_short_branch(),
    'dirty': is_dirty(),
    'origin': get_normalized_origin(),
    'deviceType': HARDWARE.get_device_type(),
  }

  # subscribe to deviceState for started state
  sm = SubMaster(['deviceState'])

  last_flush_time = time.monotonic()
  gauges = {}
  while True:
    started_prev = sm['deviceState'].started
    sm.update()

    # Update metrics
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
            cloudlog.event("unknown metric type", metric_type=metric_type)
        except Exception:
          cloudlog.event("malformed metric", metric=metric)
      except zmq.error.Again:
        break

    # flush when started state changes or after FLUSH_TIME_S
    if (time.monotonic() > last_flush_time + STATS_FLUSH_TIME_S) or (sm['deviceState'].started != started_prev):
      result = ""
      current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
      tags['started'] = sm['deviceState'].started

      for gauge_key in gauges:
        result += get_influxdb_line(f"gauge.{gauge_key}", gauges[gauge_key], current_time, tags)

      # clear intermediate data
      gauges = {}
      last_flush_time = time.monotonic()

      # check that we aren't filling up the drive
      if len(os.listdir(STATS_DIR)) < STATS_DIR_FILE_LIMIT:
        if len(result) > 0:
          stats_path = os.path.join(STATS_DIR, str(int(current_time.timestamp())))
          with atomic_write_in_dir(stats_path) as f:
            f.write(result)
      else:
        cloudlog.error("stats dir full")


if __name__ == "__main__":
  main()
else:
  statlog = StatLog()
