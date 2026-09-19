import math

EARTH_RADIUS_M = 6371009.


def wrap_angle(a: float) -> float:
  """Wrap angle to [-pi, pi]."""
  return math.remainder(a, 2 * math.pi)


def geodetic_to_local_ned(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
  """Equirectangular projection of (lat, lon) into NED meters relative to (ref_lat, ref_lon)."""
  north = math.radians(lat - ref_lat) * EARTH_RADIUS_M
  east = math.radians(lon - ref_lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
  return north, east
