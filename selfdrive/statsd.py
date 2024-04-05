#!/usr/bin/env python3
import os
import zmq
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import NoReturn

from openpilot.common.params import Params
from cereal.messaging import SubMaster
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE
from openpilot.common.file_helpers import atomic_write_in_dir
from openpilot.system.version import get_build_metadata
from openpilot.system.loggerd.config import STATS_DIR_FILE_LIMIT, STATS_SOCKET, STATS_FLUSH_TIME_S


class METRIC_TYPE:
  GAUGE = 'g'
  SAMPLE = 'sa'

class StatLog:
  def __init__(self):
    self.pid = None
    self.zctx = None
    self.sock = None

  def connect(self) -> None:
    self.zctx = zmq.Context()
    self.sock = self.zctx.socket(zmq.PUSH)
    self.sock.setsockopt(zmq.LINGER, 10)
    self.sock.connect(STATS_SOCKET)
    self.pid = os.getpid()

  def __del__(self):
    if self.sock is not None:
      self.sock.close()
    if self.zctx is not None:
      self.zctx.term()

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

  # Samples will be recorded in a buffer and at aggregation time,
  # statistical properties will be logged (mean, count, percentiles, ...)
  def sample(self, name: str, value: float):
    self._send(f"{name}:{value}|{METRIC_TYPE.SAMPLE}")


def main() -> NoReturn:
  dongle_id = Params().get("DongleId", encoding='utf-8')
  def get_influxdb_line(measurement: str, value: float | dict[str, float],  timestamp: datetime, tags: dict) -> str:
    res = f"{measurement}"
    for k, v in tags.items():
      res += f",{k}={str(v)}"
    res += " "

    if isinstance(value, float):
      value = {'value': value}

    for k, v in value.items():
      res += f"{k}={v},"

    res += f"dongle_id=\"{dongle_id}\" {int(timestamp.timestamp() * 1e9)}\n"
    return res

  # open statistics socket
  ctx = zmq.Context.instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind(STATS_SOCKET)

  STATS_DIR = Paths.stats_root()

  # initialize stats directory
  Path(STATS_DIR).mkdir(parents=True, exist_ok=True)

  build_metadata = get_build_metadata()

  # initialize tags
  tags = {
    'started': False,
    'version': build_metadata.openpilot.version,
    'branch': build_metadata.channel,
    'dirty': build_metadata.openpilot.is_dirty,
    'origin': build_metadata.openpilot.git_normalized_origin,
    'deviceType': HARDWARE.get_device_type(),
  }

  # subscribe to deviceState for started state
  sm = SubMaster(['deviceState'])

  idx = 0
  last_flush_time = time.monotonic()
  gauges = {}
  samples: dict[str, list[float]] = defaultdict(list)
  try:
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
            metric_value = float(metric.split('|')[0].split(':')[1])

            if metric_type == METRIC_TYPE.GAUGE:
              gauges[metric_name] = metric_value
            elif metric_type == METRIC_TYPE.SAMPLE:
              samples[metric_name].append(metric_value)
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

        for key, value in gauges.items():
          result += get_influxdb_line(f"gauge.{key}", value, current_time, tags)

        for key, values in samples.items():
          values.sort()
          sample_count = len(values)
          sample_sum = sum(values)

          stats = {
            'count': sample_count,
            'min': values[0],
            'max': values[-1],
            'mean': sample_sum / sample_count,
          }
          for percentile in [0.05, 0.5, 0.95]:
            value = values[int(round(percentile * (sample_count - 1)))]
            stats[f"p{int(percentile * 100)}"] = value

          result += get_influxdb_line(f"sample.{key}", stats, current_time, tags)

        # clear intermediate data
        gauges.clear()
        samples.clear()
        last_flush_time = time.monotonic()

        # check that we aren't filling up the drive
        if len(os.listdir(STATS_DIR)) < STATS_DIR_FILE_LIMIT:
          if len(result) > 0:
            stats_path = os.path.join(STATS_DIR, f"{current_time.timestamp():.0f}_{idx}")
            with atomic_write_in_dir(stats_path) as f:
              f.write(result)
            idx += 1
        else:
          cloudlog.error("stats dir full")
  finally:
    sock.close()
    ctx.term()


if __name__ == "__main__":
  main()
else:
  statlog = StatLog()
