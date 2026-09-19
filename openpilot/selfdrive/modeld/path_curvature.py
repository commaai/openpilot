from __future__ import annotations

import numpy as np

PATH_LOOKAHEAD_METERS = 3.0


def curvature_from_path(positions: np.ndarray) -> float | None:
  """Aim at a point 3 meters along the path, starting at the vehicle origin.

  Coordinates are x forward, y right. Return None if the path cannot supply a
  usable forward target. This is a pure-pursuit steering target, not local path curvature.
  """
  points = np.asarray(positions)
  if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
    return None
  if not np.isfinite(points[:, :2]).all():
    return None

  previous = np.zeros(2)
  distance = 0.0
  for point in points[:, :2]:
    segment_length = float(np.linalg.norm(point - previous))
    if segment_length > 0 and distance + segment_length >= PATH_LOOKAHEAD_METERS:
      fraction = (PATH_LOOKAHEAD_METERS - distance) / segment_length
      x, y = previous + fraction * (point - previous)
      if x <= 0:
        return None
      return float(2 * y / (x * x + y * y))
    distance += segment_length
    previous = point

  return None
