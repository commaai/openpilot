import os
import numpy as np

from openpilot.common.realtime import DT_MDL

MAP_PATH = os.path.join(os.path.dirname(__file__), 'fleet_speed_map.npz')
CELL_DEG = 3e-4
MATCH_RADIUS_M = 10.0
MAX_BEARING_DIFF_DEG = 35.0
MAX_GPS_AGE_S = 1.5
MAX_GPS_ACC_M = 15.0
MIN_GPS_SPEED = 3.0
MIN_PATH_DIST_M = 5.0
A_MAP_MIN = -1.5
HOLD_M = 400.0
HOLD_MAX_TURN_DEG = 45.0
HOLD_RELEASE_PER_M = 0.02
M_PER_DEG_LAT = 111_320.0
GPS_SERVICES = ('gpsLocationExternal', 'gpsLocation')


def cell_keys(lat, lng):
  return np.floor(np.asarray(lat) / CELL_DEG).astype(np.int64) * 10_000_000 + np.floor(np.asarray(lng) / CELL_DEG).astype(np.int64)


def angle_diff(a, b):
  return abs((a - b + 180.0) % 360.0 - 180.0)


class FleetSpeedMap:
  # 5 m bins of smoothed free-flow 85th percentile human speed, keyed by position + travel bearing
  def __init__(self, path=MAP_PATH):
    d = np.load(path)
    lat, lng = d['lat'] / 1e6, d['lng'] / 1e6
    order = np.argsort(cell_keys(lat, lng), kind='stable')
    self.lat, self.lng = lat[order], lng[order]
    self.bearing = d['bearing'][order] * (360.0 / 256.0)
    self.speed = d['speed'][order] * 0.25
    keys = cell_keys(self.lat, self.lng)
    self.keys, self.starts = np.unique(keys, return_index=True)
    self.ends = np.r_[self.starts[1:], len(keys)]

  def lookup(self, lat, lng, bearing):
    # speed of the nearest same-direction bin within MATCH_RADIUS_M, else nan
    ci, cj = int(np.floor(lat / CELL_DEG)), int(np.floor(lng / CELL_DEG))
    keys = np.array([(ci + di) * 10_000_000 + cj + dj for di in (-1, 0, 1) for dj in (-1, 0, 1)])
    pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
    pos = pos[self.keys[pos] == keys]
    if not len(pos):
      return np.nan
    idx = np.concatenate([np.arange(self.starts[p], self.ends[p]) for p in pos])
    dn = (self.lat[idx] - lat) * M_PER_DEG_LAT
    de = (self.lng[idx] - lng) * M_PER_DEG_LAT * np.cos(np.radians(lat))
    dist = np.hypot(dn, de)
    ok = (dist < MATCH_RADIUS_M) & (angle_diff(self.bearing[idx], bearing) < MAX_BEARING_DIFF_DEG)
    if not ok.any():
      return np.nan
    return float(self.speed[idx[ok][np.argmin(dist[ok])]])


class FleetSpeedCap:
  def __init__(self, path=MAP_PATH, dt=DT_MDL):
    self.map = FleetSpeedMap(path) if os.path.exists(path) else None
    self.dt = dt
    self.v_here = np.nan      # speed the cap uses at the car's position (mapped or gap-filled)
    self.v_target = np.nan    # speed of the path point that sets the cap
    self.gap_filled = False
    self.last_v = np.nan      # last mapped speed at the car's position, held through unmapped gaps
    self.last_bearing = 0.0
    self.dist_since = 0.0

  def _gps(self, sm):
    for s in GPS_SERVICES:
      if s not in getattr(sm, 'services', ()) or sm.recv_frame[s] == 0:
        continue
      g = sm[s]
      age = (sm.logMonoTime['modelV2'] - sm.logMonoTime[s]) * 1e-9
      if g.hasFix and 0 <= age < MAX_GPS_AGE_S and g.horizontalAccuracy < MAX_GPS_ACC_M and g.speed > MIN_GPS_SPEED:
        return g, age
    return None, 0.0

  def update(self, sm, v_ego):
    # accel needed to be at or below the cap at every point of the model path, None when unconstrained
    self.v_here, self.v_target, self.gap_filled = np.nan, np.nan, False
    self.dist_since += v_ego * self.dt
    if self.map is None:
      return None
    gps, gps_age = self._gps(sm)
    if gps is None:
      return None

    b0 = np.radians(gps.bearingDeg)
    x = np.asarray(sm['modelV2'].position.x) + gps.speed * gps_age
    y = np.asarray(sm['modelV2'].position.y)
    if len(x) < 2:
      return None
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] + gps.speed * gps_age
    east = x * np.sin(b0) + y * np.cos(b0)
    north = x * np.cos(b0) - y * np.sin(b0)
    lats = gps.latitude + north / M_PER_DEG_LAT
    lngs = gps.longitude + east / (M_PER_DEG_LAT * np.cos(np.radians(gps.latitude)))
    bearings = np.degrees(b0 + np.arctan2(np.gradient(y), np.gradient(x))) % 360.0
    v_raw = np.array([self.map.lookup(lats[i], lngs[i], bearings[i]) for i in range(len(x))])
    # a slow point only counts when the next path point agrees, so one mismatched bin can't brake the car
    v_path = np.r_[np.maximum(v_raw[:-1], v_raw[1:]), v_raw[-1]]

    # speed at the car: mapped, else interpolated from the last mapped speed behind to the next one ahead on this road
    if not np.isnan(v_path[0]):
      self.v_here = v_path[0]
      self.last_v, self.last_bearing, self.dist_since = v_path[0], gps.bearingDeg, 0.0
    elif not np.isnan(self.last_v) and self.dist_since < HOLD_M and angle_diff(gps.bearingDeg, self.last_bearing) < HOLD_MAX_TURN_DEG:
      ahead = np.flatnonzero(~np.isnan(v_path))
      if len(ahead):
        k = ahead[0]
        self.v_here = self.last_v + (v_path[k] - self.last_v) * self.dist_since / (self.dist_since + s[k])
      else:
        self.v_here = self.last_v + HOLD_RELEASE_PER_M * self.dist_since
      self.gap_filled = True
    else:
      self.last_v = np.nan

    a_map, v_bind = None, np.nan
    for i, v in enumerate(v_path):
      if i == 0 and np.isnan(v):
        v = self.v_here
      if np.isnan(v) or v >= v_ego:
        continue
      a = (v - v_ego) if s[i] < MIN_PATH_DIST_M else (v ** 2 - v_ego ** 2) / (2.0 * s[i])
      if a_map is None or a < a_map:
        a_map, v_bind = a, v
    self.v_target = v_bind
    return None if a_map is None else max(a_map, A_MAP_MIN)
