"""Planar geometry for routes, and route-relative map pose and progress tracking."""
import numpy as np

METERS_PER_DEGREE = 2 * np.pi * 6371008.8 / 360  # mean Earth radius, as Mapbox measures distances

# Training renders the map on the route rather than at the measured position:
# the ego is projected onto the route, the frame is centered there and headed along the route's tangent.
PROJECTION_WINDOW = (-30.0, 100.0)  # meters around the previous progress to search; a loop cannot capture the ego
HEADING_HALF_WINDOW = 50.0  # meters; the heading is the chord of the route this far behind and ahead


def local_meters(points, origin) -> np.ndarray:
  """Equirectangular east/north meters of (longitude, latitude) points around an origin."""
  origin = np.asarray(origin, dtype=np.float64)
  scale = np.array([METERS_PER_DEGREE * np.cos(np.radians(origin[1])), METERS_PER_DEGREE])
  return (np.asarray(points, dtype=np.float64) - origin) * scale


def nearest_segment(starts: np.ndarray, ends: np.ndarray, point) -> tuple[int, float, float]:
  """(index, fraction along it, distance) of the closest spot on the segments, in the inputs' units."""
  ab = ends - starts
  t = np.clip(((point - starts) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0, 1)
  d = np.linalg.norm(starts + t[:, None] * ab - point, axis=1)
  i = int(d.argmin())
  return i, float(t[i]), float(d[i])


def cumulative_distance(route: np.ndarray) -> np.ndarray:
  """Meters along [(longitude, latitude), ...]."""
  (lon1, lat1), (lon2, lat2) = route[:-1].T, route[1:].T
  steps = np.hypot((lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2)), lat2 - lat1) * METERS_PER_DEGREE
  return np.r_[0.0, np.cumsum(steps)]


def along_path(path: np.ndarray, point) -> tuple[float, float]:
  """(meters along a (longitude, latitude) path to a point, and the point's distance from the path):
  the path up to the start of the nearest segment, then straight to the point."""
  local, p = local_meters(path, path[0]), local_meters(point, path[0])
  segment, _, lateral = nearest_segment(local[:-1], local[1:], p)
  return float(cumulative_distance(path)[segment] + np.linalg.norm(local_meters(point, path[segment]))), lateral


class RouteTracker:
  """Progress of the ego along a route, in meters, and the map pose at that progress."""

  def __init__(self, route: np.ndarray):
    self.route = np.asarray(route, dtype=np.float64).reshape((-1, 2))
    self.path = local_meters(self.route, self.route[0])
    self.distance = cumulative_distance(self.route)
    self.progress = -np.inf  # the first fix may land anywhere on the route

  def project(self, longitude: float, latitude: float) -> float:
    """Advance to the nearest spot on the route within the window ahead of the previous progress."""
    near = np.flatnonzero((self.distance[1:] >= self.progress + PROJECTION_WINDOW[0]) & (self.distance[:-1] <= self.progress + PROJECTION_WINDOW[1]))
    if not len(near):  # only before the first fix, when the progress is still -inf
      near = np.arange(len(self.path) - 1)
    point = local_meters([longitude, latitude], self.route[0])
    segment, along, _ = nearest_segment(self.path[near], self.path[near + 1], point)
    segment = near[segment]
    self.progress = max(self.progress, self.distance[segment] + along * (self.distance[segment + 1] - self.distance[segment]))
    return self.progress

  def pose(self, progress: float) -> tuple[float, float, float]:
    """(longitude, latitude, bearing) on the route at a progress, headed along the +-50m chord."""
    ends = np.clip([progress - HEADING_HALF_WINDOW, progress, progress + HEADING_HALF_WINDOW], 0, self.distance[-1])
    (lon0, lon, lon1), (lat0, lat, lat1) = (np.interp(ends, self.distance, self.route[:, axis]) for axis in (0, 1))
    bearing = np.degrees(np.arctan2((lon1 - lon0) * np.cos(np.radians(lat0)), lat1 - lat0)) % 360
    return float(lon), float(lat), float(bearing)

  def snap(self, longitude: float, latitude: float) -> tuple[float, float, float]:
    return self.pose(self.project(longitude, latitude))
