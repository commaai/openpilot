"""Log extracted openpilot data into Rerun."""

from __future__ import annotations

import logging

import numpy as np

from openpilot.tools.rerun_bridge.gps import GpsPoint, build_gps_trace
from openpilot.tools.rerun_bridge.logs import LogEntry, TimelineEntry
from openpilot.tools.rerun_bridge.time_axis import ROUTE_TIMELINE, set_route_time

logger = logging.getLogger(__name__)


def log_series(rr, series: dict[str, tuple[list[float], list[float]]], min_points: int = 2) -> int:
  logged = 0
  for path, (times, values) in series.items():
    if len(times) < min_points or len(times) != len(values):
      continue
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    entity = path.lstrip("/")
    rr.send_columns(
      entity,
      indexes=[rr.TimeColumn(ROUTE_TIMELINE, duration=t)],
      columns=rr.Scalars.columns(scalars=v),
    )
    logged += 1
  return logged


def log_text_logs(rr, logs: list[LogEntry]) -> int:
  for entry in logs:
    set_route_time(rr, entry.mono_time)
    level = {
      10: rr.TextLogLevel.TRACE,
      20: rr.TextLogLevel.DEBUG,
      30: rr.TextLogLevel.INFO,
      40: rr.TextLogLevel.WARN,
      50: rr.TextLogLevel.ERROR,
    }.get(entry.level, rr.TextLogLevel.INFO)
    rr.log("logs", rr.TextLog(entry.message, level=level))
  return len(logs)


def log_timeline(rr, timeline: list[TimelineEntry]) -> None:
  for entry in timeline:
    set_route_time(rr, entry.start_time)
    rr.log("timeline", rr.TextLog(f"{entry.kind} span", level=rr.TextLogLevel.INFO))


def log_gps(rr, series: dict[str, tuple[list[float], list[float]]]) -> int:
  points: list[GpsPoint] = build_gps_trace(series)
  if len(points) < 2:
    return 0

  lats = np.array([p.lat for p in points], dtype=np.float64)
  lons = np.array([p.lon for p in points], dtype=np.float64)
  lat_lon = np.column_stack([lats, lons])
  rr.log("map/trace", rr.GeoLineStrings(lat_lon=[lat_lon]), static=True)
  rr.log("map/position", rr.GeoPoints(lat_lon=lat_lon), static=True)
  return len(points)