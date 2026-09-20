from __future__ import annotations

import numpy as np

PATH_LOOKAHEAD_METERS = 5.0


def curvature_from_path(positions: np.ndarray) -> float | None:
  """Aim up to 5 meters along the path, or at its end if it is shorter.

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
  target = points[-1, :2]
  for point in points[:, :2]:
    segment_length = float(np.linalg.norm(point - previous))
    if segment_length > 0 and distance + segment_length >= PATH_LOOKAHEAD_METERS:
      fraction = (PATH_LOOKAHEAD_METERS - distance) / segment_length
      target = previous + fraction * (point - previous)
      break
    distance += segment_length
    previous = point

  x, y = target
  target_distance_squared = float(x * x + y * y)
  if x <= 0 or target_distance_squared < 1e-6:
    return None
  return float(2 * y / target_distance_squared)
