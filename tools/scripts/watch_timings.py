#!/usr/bin/env python3
import argparse
import datetime
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal.services import SERVICE_LIST


@dataclass
class ServiceTiming:
  times: list[float] = field(default_factory=list)
  window: deque[float] = field(default_factory=lambda: deque(maxlen=100))
  valids: deque[bool] = field(default_factory=lambda: deque(maxlen=100))
  lag_events: list[tuple[float, float]] = field(default_factory=list)

  def add(self, mono_time: float, valid: bool, expected_interval: float | None, lag_threshold: float) -> None:
    if self.times:
      dt = mono_time - self.times[-1]
      self.window.append(dt)
      if expected_interval is not None and dt > lag_threshold * expected_interval:
        self.lag_events.append((mono_time, dt))

    self.times.append(mono_time)
    self.valids.append(valid)

  def intervals(self, latest_only: bool) -> np.ndarray:
    if latest_only:
      return np.array(self.window)
    return np.diff(self.times)


def format_row(name: str, timing: ServiceTiming, latest_only: bool) -> str:
  dts = timing.intervals(latest_only)
  if len(dts) == 0:
    return f"{name:25} waiting for messages"

  mean = np.mean(dts)
  hz = 1.0 / mean if mean > 0 else 0.0
  valid = all(timing.valids) if timing.valids else False
  return f"{name:25} {hz:8.2f}Hz {mean * 1e3:8.2f}ms {np.std(dts) * 1e3:8.2f}ms {np.max(dts) * 1e3:8.2f}ms {np.min(dts) * 1e3:8.2f}ms valid={valid}"


def print_lag_events(name: str, timing: ServiceTiming, printed_lags: dict[str, int]) -> None:
  start = printed_lags.get(name, 0)
  for mono_time, dt in timing.lag_events[start:]:
    print(f"{mono_time:.3f} {name} lag {dt:.3f}s", flush=True)
  printed_lags[name] = len(timing.lag_events)


def monitor_services(socket_names: list[str], print_interval: float, lag_threshold: float, lag_only: bool) -> None:
  sockets = {name: messaging.sub_sock(name, conflate=False) for name in socket_names}
  timings = {name: ServiceTiming() for name in socket_names}
  printed_lags: dict[str, int] = {}

  start_time = time.monotonic()
  last_print = start_time

  try:
    while True:
      for name, sock in sockets.items():
        for msg in messaging.drain_sock(sock):
          expected_interval = 1.0 / SERVICE_LIST[name].frequency if name in SERVICE_LIST else None
          timings[name].add(msg.logMonoTime / 1e9, msg.valid, expected_interval, lag_threshold)

      now = time.monotonic()
      if now - last_print < print_interval:
        time.sleep(0.01)
        continue

      if not lag_only:
        print(flush=True)
        print(f"{'service':25} {'freq':>10} {'mean':>10} {'std':>10} {'max':>10} {'min':>10} valid", flush=True)
        for name in socket_names:
          print(format_row(name, timings[name], latest_only=True), flush=True)

      for name in socket_names:
        print_lag_events(name, timings[name], printed_lags)

      last_print = now
  except KeyboardInterrupt:
    print("\n", flush=True)
    print("=" * 5, "timing summary", "=" * 5, flush=True)
    print(f"{'service':25} {'freq':>10} {'mean':>10} {'std':>10} {'max':>10} {'min':>10} valid", flush=True)
    for name in socket_names:
      print(format_row(name, timings[name], latest_only=False), flush=True)
    print("=" * 5, datetime.timedelta(seconds=time.monotonic() - start_time), "=" * 5, flush=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Check live service timing, frequency, validity, and lag")
  parser.add_argument("socket", nargs="*", default=["carState"], help="service/socket name")
  parser.add_argument("--lag-threshold", type=float, default=10.0, help="report intervals above this multiple of the expected service interval")
  parser.add_argument("--lag-only", action="store_true", help="only print lag events")
  parser.add_argument("--print-interval", type=float, default=1.0, help="seconds between table updates")
  args = parser.parse_args()

  monitor_services(args.socket, args.print_interval, args.lag_threshold, args.lag_only)
