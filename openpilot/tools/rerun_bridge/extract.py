"""Extract jotpluggler-compatible scalar paths from cereal events."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.lib.log_time_series import msgs_to_time_series


@dataclass
class CanFrame:
  service: str
  bus: int
  address: int
  bus_time: int
  data: bytes
  mono_time: float


@dataclass
class SeriesStore:
  series: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
  enum_info: dict[str, list[str]] = field(default_factory=dict)
  can_frames: list[CanFrame] = field(default_factory=list)

  def append(self, path: str, t: float, value: float) -> None:
    bucket = self.series.get(path)
    if bucket is None:
      bucket = ([], [])
      self.series[path] = bucket
    bucket[0].append(t)
    bucket[1].append(value)

  def set_series(self, path: str, times: list[float], values: list[float]) -> None:
    if len(times) >= 2 and len(times) == len(values):
      self.series[path] = (times, values)


def _scalar_values(values) -> list[float] | None:
  if isinstance(values, (bool, np.bool_)):
    return [1.0 if values else 0.0]
  if isinstance(values, (int, float, np.integer, np.floating)):
    return [float(values)]
  if isinstance(values, np.ndarray):
    if values.dtype == object:
      return None
    flat = values.reshape(-1)
    if flat.size == 0:
      return None
    try:
      return [float(x) for x in flat.tolist()]
    except (TypeError, ValueError):
      return None
  if isinstance(values, list):
    if not values:
      return None
    try:
      return [float(x) for x in values]
    except (TypeError, ValueError):
      return None
  return None


def _flatten_group(prefix: str, group: dict, out: dict[str, tuple[list[float], list[float]]]) -> None:
  times = [float(t) for t in group.get("t", [])]
  if len(times) < 2:
    return
  base = f"/{prefix}"

  for key, values in group.items():
    if key == "t":
      continue
    path = f"{base}/{key}"

    if isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[0] == len(times):
      width = values.shape[1]
      if width <= 32:
        for index in range(min(width, 32)):
          col = values[:, index]
          try:
            out[f"{path}/{index}"] = (times, [float(v) for v in col.tolist()])
          except (TypeError, ValueError):
            pass
      continue

    if isinstance(values, np.ndarray) and values.dtype == object and values.shape[0] == len(times):
      sample = next((row for row in values if row is not None and getattr(row, "size", 0) > 0), None)
      if sample is not None and getattr(sample, "ndim", 0) == 1 and sample.size <= 32:
        cols: list[list[float]] = [[] for _ in range(sample.size)]
        ok = True
        for row in values:
          if row is None or getattr(row, "size", 0) != sample.size:
            ok = False
            break
          for index, item in enumerate(row.tolist()):
            cols[index].append(float(item))
        if ok:
          for index, col in enumerate(cols):
            out[f"{path}/{index}"] = (times, col)
        continue

    scalar = _scalar_values(values)
    if scalar is None:
      continue
    if len(scalar) == len(times):
      out[path] = (times, scalar)
    elif len(scalar) == 1:
      out[path] = (times, scalar * len(times))


def _extract_meta(messages) -> dict[str, tuple[list[float], list[float]]]:
  out: dict[str, tuple[list[float], list[float]]] = {}
  buckets: dict[str, tuple[list[float], list[float], list[float], list[float]]] = {}

  for event in messages:
    try:
      which = event.which()
    except Exception:
      continue
    if which == "sentinel":
      continue
    tm = float(event.logMonoTime) / 1e9
    base = f"/{which}"
    bucket = buckets.get(base)
    if bucket is None:
      bucket = ([], [], [], [])
      buckets[base] = bucket
    bucket[0].append(tm)
    bucket[1].append(1.0 if event.valid else 0.0)
    bucket[2].append(float(event.logMonoTime))
    bucket[3].append(tm)

    if which in {"can", "sendcan"}:
      continue

  for base, (t, valid, mono, seconds) in buckets.items():
    if len(t) < 2:
      continue
    out[f"{base}/valid"] = (t, valid)
    out[f"{base}/logMonoTime"] = (t, mono)
    out[f"{base}/t"] = (t, seconds)
  return out


def _extract_can(messages, store: SeriesStore) -> None:
  for event in messages:
    try:
      which = event.which()
    except Exception:
      continue
    if which not in {"can", "sendcan"}:
      continue
    tm = float(event.logMonoTime) / 1e9
    for msg in getattr(event, which):
      bus = int(msg.src)
      address = int(msg.address)
      dat = bytes(msg.dat)
      store.can_frames.append(CanFrame(which, bus, address, int(msg.deprecated.busTime), dat, tm))
      store.append(f"/{which}/{bus}/{address}/busTime", tm, float(msg.deprecated.busTime))
      store.append(f"/{which}/{bus}/{address}/present", tm, 1.0)


class EventExtractor:
  def __init__(self, include_deprecated: bool = True):
    self.include_deprecated = include_deprecated

  def process_events(self, messages) -> SeriesStore:
    store = SeriesStore()
    groups = msgs_to_time_series(messages)

    for prefix, group in groups.items():
      if not self.include_deprecated and "DEPRECATED" in prefix:
        continue
      _flatten_group(prefix, group, store.series)

    store.series.update(_extract_meta(messages))
    _extract_can(messages, store)
    return store


def finalize_series(store: SeriesStore, include_deprecated: bool = True) -> dict[str, tuple[list[float], list[float]]]:
  out: dict[str, tuple[list[float], list[float]]] = {}
  for path, (times, values) in store.series.items():
    if len(times) < 2 or len(times) != len(values):
      continue
    if not all(math.isfinite(t) for t in times):
      continue
    if not include_deprecated and "DEPRECATED" in path:
      continue
    out[path] = (times, values)
  return out