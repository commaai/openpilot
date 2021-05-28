from enum import Enum
import numpy as np


R = 6373000.0  # approximate radius of earth in mts


def vectors(points):
  """Provides a array of vectors on cartesian space (x, y).
     Each vector represents the path from a point in `points` to the next.
     `points` must by a (N, 2) array of [lat, lon] pairs in radians.
  """
  latA = points[:-1, 0]
  latB = points[1:, 0]
  delta = np.diff(points, axis=0)
  dlon = delta[:, 1]

  x = np.sin(dlon) * np.cos(latB)
  y = np.cos(latA) * np.sin(latB) - (np.sin(latA) * np.cos(latB) * np.cos(dlon))

  return np.column_stack((x, y))


def ref_vectors(ref, points):
  """Provides a array of vectors on cartesian space (x, y).
     Each vector represents the path from ref to a point in `points`.
     `points` must by a (N, 2) array of [lat, lon] pairs in radians.
  """
  latA = ref[0]
  latB = points[:, 0]
  delta = points - ref
  dlon = delta[:, 1]

  x = np.sin(dlon) * np.cos(latB)
  y = np.cos(latA) * np.sin(latB) - (np.sin(latA) * np.cos(latB) * np.cos(dlon))

  return np.column_stack((x, y))


def bearing_to_points(point, points):
  """Calculate the bearings (angle from true north clockwise) of the vectors between `point` and each
  one of the entries in `points`. Both `point` and `points` elements are 2 element arrays containing a latitud,
  longitude pair in radians.
  """
  delta = points - point
  x = np.sin(delta[:, 1]) * np.cos(points[:, 0])
  y = np.cos(point[0]) * np.sin(points[:, 0]) - (np.sin(point[0]) * np.cos(points[:, 0]) * np.cos(delta[:, 1]))
  return np.arctan2(x, y)


def distance_to_points(point, points):
  """Calculate the distance of the vectors between `point` and each one of the entries in `points`.
  Both `point` and `points` elements are 2 element arrays containing a latitud, longitude pair in radians.
  """
  delta = points - point
  a = np.sin(delta[:, 0] / 2)**2 + np.cos(point[0]) * np.cos(points[:, 0]) * np.sin(delta[:, 1] / 2)**2
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return c * R


class DIRECTION(Enum):
  NONE = 0
  AHEAD = 1
  BEHIND = 2
  FORWARD = 3
  BACKWARD = 4
