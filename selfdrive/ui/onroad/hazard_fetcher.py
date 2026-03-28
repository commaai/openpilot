import math
import time
import threading
from dataclasses import dataclass, field

import requests

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.onroad.hazard_scoring import HazardScore, hazard_score_from_api

BASE_URL = "https://roadpass.jpadams.xyz"
TIMEOUT = 10.0

EARTH_RADIUS_M = 6_371_000.0

# Refresh rate on the same road/heading.
SAME_ROAD_INTERVAL = 60.0
# Minimum interval once a direction change has been detected.
DIRECTION_CHANGE_INTERVAL = 5.0
# Bearing shift (degrees) that counts as a direction change.
DIRECTION_CHANGE_THRESHOLD = 45.0
# How long (seconds) to stay in fast-refresh mode after a turn.
DIRECTION_CHANGE_WINDOW = 15.0

# Speed buckets: (max_speed_ms, fetch_radius_m, warn_distance_m)
# warn_distance_m targets roughly 20-27 seconds of lead-time at each bucket's
# typical mid-speed.
_SPEED_BUCKETS: list[tuple[float, int, int]] = [
    (4.5,          500,  100),   # < 10 mph  — city crawl
    (11.2,         800,  200),   # 10–25 mph — city
    (20.1,        1609,  400),   # 25–45 mph — suburban
    (29.1,        2500,  600),   # 45–65 mph — road
    (math.inf,    4000,  900),   # > 65 mph  — highway
]


# ── Geo helpers ────────────────────────────────────────────────────────────────

def _bearing_delta(a: float, b: float) -> float:
  """Smallest unsigned angle between two compass bearings (0–180°)."""
  delta = abs(a - b) % 360
  return delta if delta <= 180 else 360 - delta


def _speed_to_bucket(speed_ms: float) -> tuple[int, int]:
  """Return (fetch_radius_m, warn_distance_m) for the given speed."""
  for max_spd, radius, warn in _SPEED_BUCKETS:
    if speed_ms < max_spd:
      return radius, warn
  return _SPEED_BUCKETS[-1][1], _SPEED_BUCKETS[-1][2]


def _project_position(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
  """
  Spherical-Earth forward projection: given a start point and a bearing +
  distance, return the destination (lat, lon).
  """
  d = distance_m / EARTH_RADIUS_M
  lat1 = math.radians(lat)
  lon1 = math.radians(lon)
  brng = math.radians(bearing_deg)

  lat2 = math.asin(
    math.sin(lat1) * math.cos(d) +
    math.cos(lat1) * math.sin(d) * math.cos(brng)
  )
  lon2 = lon1 + math.atan2(
    math.sin(brng) * math.sin(d) * math.cos(lat1),
    math.cos(d) - math.sin(lat1) * math.sin(lat2),
  )
  return math.degrees(lat2), math.degrees(lon2)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Great-circle distance in metres between two (lat, lon) points."""
  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = (
    math.sin(dlat / 2) ** 2 +
    math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
  )
  return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _haversine_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Forward azimuth (0–360°) from point 1 to point 2."""
  dlon = math.radians(lon2 - lon1)
  lat1r = math.radians(lat1)
  lat2r = math.radians(lat2)
  x = math.sin(dlon) * math.cos(lat2r)
  y = (math.cos(lat1r) * math.sin(lat2r) -
       math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
  return (math.degrees(math.atan2(x, y)) + 360) % 360


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class HazardAhead:
  """
  A single upcoming hazard, stored with its absolute (lat, lon) so that
  distance and bearing stay accurate as the device moves after a fetch.
  """
  event_id: str
  lat: float
  lon: float
  accel_ms2: float
  response_summary: dict = field(default_factory=dict)
  score: HazardScore | None = None

  @classmethod
  def from_api(cls, data: dict, device_lat: float, device_lon: float) -> 'HazardAhead':
    """
    The API returns distance_m + bearing_deg relative to the device's position
    at fetch time. Project those back to an absolute position so we can
    recompute distance live on subsequent frames.
    """
    lat, lon = _project_position(device_lat, device_lon, data['bearing_deg'], data['distance_m'])
    return cls(
      event_id=data['event_id'],
      lat=lat,
      lon=lon,
      accel_ms2=data.get('accel_ms2', 0.0),
      response_summary=data.get('response_summary', {}),
      score=hazard_score_from_api(data),
    )

  def distance_m(self, device_lat: float, device_lon: float) -> float:
    return _haversine(device_lat, device_lon, self.lat, self.lon)

  def bearing_deg(self, device_lat: float, device_lon: float) -> float:
    return _haversine_bearing(device_lat, device_lon, self.lat, self.lon)


# ── Fetcher ────────────────────────────────────────────────────────────────────

class HazardFetcher:
  """
  Background service that polls GET /hazards/ahead and keeps results in memory.

  Refresh policy:
    - Same road / steady heading:  at most once per 60 s
    - After a direction change:    as often as every 5 s for 15 s, then
                                   falls back to the 60 s cadence

  GPS state is fed in from the main thread via update_gps(); the worker thread
  only reads it under a lock and never touches ui_state directly.
  """

  def __init__(self):
    self._session = requests.Session()
    self._lock = threading.Lock()

    # Written by the main thread, read by the worker.
    self._gps: tuple[float, float, float, float, bool] | None = None  # lat, lon, bearing, speed_ms, has_fix
    self._prev_bearing: float | None = None
    self._last_direction_change: float = -DIRECTION_CHANGE_WINDOW  # allow fetch on first call

    self._last_fetch_time: float = 0.0
    self._hazards: list[HazardAhead] = []

    self._thread = threading.Thread(target=self._worker, daemon=True)
    self._thread.start()

  def update_gps(self, lat: float, lon: float, bearing_deg: float,
                 speed_ms: float, has_fix: bool) -> None:
    """
    Call this from the main (render) thread each frame. Detects direction
    changes and updates the GPS snapshot used by the worker.
    """
    now = time.monotonic()
    with self._lock:
      if self._prev_bearing is not None:
        if _bearing_delta(bearing_deg, self._prev_bearing) > DIRECTION_CHANGE_THRESHOLD:
          self._last_direction_change = now
      self._prev_bearing = bearing_deg
      self._gps = (lat, lon, bearing_deg, speed_ms, has_fix)

  def get_hazards(self) -> list[HazardAhead]:
    """Snapshot of cached hazards. Safe to call from any thread."""
    with self._lock:
      return list(self._hazards)

  # ── private ──────────────────────────────────────────────────────────────────

  def _worker(self) -> None:
    while True:
      time.sleep(1.0)
      self._maybe_fetch()

  def _maybe_fetch(self) -> None:
    now = time.monotonic()
    with self._lock:
      gps = self._gps
      last_dir_change = self._last_direction_change

    if gps is None:
      return

    lat, lon, bearing, speed_ms, has_fix = gps
    if not has_fix:
      return

    elapsed = now - self._last_fetch_time
    recently_turned = (now - last_dir_change) < DIRECTION_CHANGE_WINDOW
    min_interval = DIRECTION_CHANGE_INTERVAL if recently_turned else SAME_ROAD_INTERVAL

    if elapsed < min_interval:
      return

    radius_m, _ = _speed_to_bucket(speed_ms)
    self._fetch(lat, lon, bearing, radius_m)
    self._last_fetch_time = now

  def _fetch(self, lat: float, lon: float, bearing: float, radius_m: int) -> None:
    try:
      resp = self._session.get(
        f"{BASE_URL}/hazards/ahead",
        params={"lat": lat, "lon": lon, "bearing": bearing, "radius_m": radius_m},
        timeout=TIMEOUT,
      )
      resp.raise_for_status()
      data = resp.json()
      hazards = [HazardAhead.from_api(h, lat, lon) for h in data.get("hazards", [])]
      cloudlog.info(f"HazardFetcher: {len(hazards)} hazard(s), radius={radius_m}m")
      with self._lock:
        self._hazards = hazards
    except Exception as e:
      cloudlog.error(f"HazardFetcher: fetch failed: {e}")
