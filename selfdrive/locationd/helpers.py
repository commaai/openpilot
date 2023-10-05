import numpy as np
from openpilot.common.numpy_helpers import NPQueue


class PointBuckets:
  def __init__(self, x_bounds, min_points, min_points_total, points_per_bucket, rowsize):
    self.x_bounds = x_bounds
    self.buckets = {bounds: NPQueue(maxlen=points_per_bucket, rowsize=rowsize) for bounds in x_bounds}
    self.buckets_min_points = dict(zip(x_bounds, min_points, strict=True))
    self.min_points_total = min_points_total

  def bucket_lengths(self):
    return [len(v) for v in self.buckets.values()]

  def __len__(self):
    return sum(self.bucket_lengths())

  def is_valid(self):
    individual_buckets_valid = all(len(v) >= min_pts for v, min_pts in zip(self.buckets.values(), self.buckets_min_points.values(), strict=True))
    total_points_valid = self.__len__() >= self.min_points_total
    return individual_buckets_valid and total_points_valid

  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break

  def get_points(self, num_points=None):
    points = np.vstack([x.arr for x in self.buckets.values()])
    if num_points is None:
      return points
    return points[np.random.choice(np.arange(len(points)), min(len(points), num_points), replace=False)]

  def load_points(self, points):
    for x, y in points:
      self.add_point(x, y)
