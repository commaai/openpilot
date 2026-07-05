"""Build GPS traces for Rerun map views."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GpsPoint:
  time: float
  lat: float
  lon: float
  bearing: float = 0.0


def _sample_at(series: dict[str, tuple[list[float], list[float]]], path: str, t: float) -> float | None:
  bucket = series.get(path)
  if bucket is None:
    return None
  times, values = bucket
  if len(times) < 2:
    return None
  # nearest sample
  best_i = min(range(len(times)), key=lambda i: abs(times[i] - t))
  return values[best_i]


def build_gps_trace(series: dict[str, tuple[list[float], list[float]]]) -> list[GpsPoint]:
  lat_path = "/gpsLocationExternal/latitude"
  lon_path = "/gpsLocationExternal/longitude"
  fix_path = "/gpsLocationExternal/hasFix"
  bearing_path = "/gpsLocationExternal/bearingDeg"

  lat = series.get(lat_path)
  lon = series.get(lon_path)
  fix = series.get(fix_path)
  if lat is None or lon is None or fix is None:
    return []

  lat_t, lat_v = lat
  lon_t, lon_v = lon
  fix_t, fix_v = fix
  count = min(len(lat_t), len(lon_t), len(fix_t))
  points: list[GpsPoint] = []

  for i in range(count):
    if fix_v[i] < 0.5:
      continue
    latitude = lat_v[i]
    longitude = lon_v[i]
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
      continue
    tm = lat_t[i]
    bearing = _sample_at(series, bearing_path, tm) or 0.0
    points.append(GpsPoint(time=tm, lat=latitude, lon=longitude, bearing=bearing))
  return points