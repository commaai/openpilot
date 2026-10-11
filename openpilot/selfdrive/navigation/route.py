"""Route-relative map pose and progress tracking."""
import numpy as np

MAX_LATITUDE = 85.05112878


def route_coordinates(route) -> np.ndarray:
  """A navRoute message as (longitude, latitude) rows."""
  return np.array([(c.longitude, c.latitude) for c in route.coordinates], dtype=np.float64).reshape((-1, 2))


# Training renders the map on the route rather than at the measured position:
# the ego is projected onto the route, the frame is centred there and headed along the route's tangent.
PROJECTION_WINDOW = (-30.0, 100.0)  # meters around the previous progress to search; a loop cannot capture the ego
HEADING_HALF_WINDOW = 50.0  # meters; the heading is the chord of the route this far behind and ahead


def cumulative_distance(route: np.ndarray) -> np.ndarray:
  """Haversine meters along [(longitude, latitude), ...]."""
  points = np.radians(route)
  (lon1, lat1), (lon2, lat2) = points[:-1].T, points[1:].T
  h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
  return np.r_[0.0, np.cumsum(2 * 6371008.8 * np.arctan2(np.sqrt(h), np.sqrt(1 - h)))]


class RouteTracker:
  """Progress of the ego along a route, in meters, and the map pose at that progress."""

  def __init__(self, route: np.ndarray):
    self.route = np.asarray(route, dtype=np.float64).reshape((-1, 2))
    self.distance = cumulative_distance(self.route) if len(self.route) >= 2 else np.zeros(len(self.route))
    self.progress = -np.inf  # the first fix may land anywhere on the route

  def project(self, longitude: float, latitude: float) -> float:
    """Advance to the nearest spot on the route within the window ahead of the previous progress."""
    if len(self.route) < 2:
      raise ValueError("Route needs at least two coordinates")
    origin = self.route[0]
    scale = np.array([111320.0 * np.cos(np.radians(origin[1])), 110540.0])
    path, point = (self.route - origin) * scale, (np.array([longitude, latitude]) - origin) * scale
    near = np.flatnonzero((self.distance[1:] >= self.progress + PROJECTION_WINDOW[0]) & (self.distance[:-1] <= self.progress + PROJECTION_WINDOW[1]))
    if not len(near):
      near = np.arange(len(path) - 1)
    a, ab = path[near], path[near + 1] - path[near]
    along = np.clip(((point - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0, 1)
    best = int(((a + along[:, None] * ab - point) ** 2).sum(axis=1).argmin())
    segment = near[best]
    self.progress = max(self.progress, self.distance[segment] + along[best] * (self.distance[segment + 1] - self.distance[segment]))
    return self.progress

  def pose(self, progress: float) -> tuple[float, float, float]:
    """(longitude, latitude, bearing) on the route at a progress, headed along the +-50m chord."""
    ends = np.clip([progress - HEADING_HALF_WINDOW, progress, progress + HEADING_HALF_WINDOW], 0, self.distance[-1])
    (lon0, lon, lon1), (lat0, lat, lat1) = (np.interp(ends, self.distance, self.route[:, axis]) for axis in (0, 1))
    bearing = np.degrees(np.arctan2((lon1 - lon0) * np.cos(np.radians(lat0)), lat1 - lat0)) % 360
    return float(lon), float(lat), float(bearing)

  def snap(self, longitude: float, latitude: float) -> tuple[float, float, float]:
    return self.pose(self.project(longitude, latitude))
