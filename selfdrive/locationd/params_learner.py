import numpy as np


class NPQueue:
  def __init__(self, maxlen, rowsize):
    self.maxlen = maxlen
    self.arr = np.empty((0, rowsize))

  def __len__(self):
    return len(self.arr)

  def append(self, pt):
    if len(self.arr) < self.maxlen:
      self.arr = np.append(self.arr, [pt], axis=0)
    else:
      self.arr[:-1] = self.arr[1:]
      self.arr[-1] = pt


class PointBuckets:
  def __init__(self, x_bounds, min_points, min_points_total, points_per_bucket):
    self.x_bounds = x_bounds
    self.x_size = len(x_bounds[0])
    self.buckets = {bounds: NPQueue(maxlen=points_per_bucket, rowsize=self.x_size + 2) for bounds in x_bounds}
    self.buckets_min_points = dict(zip(x_bounds, min_points, strict=True))
    self.min_points_total = min_points_total

  def bucket_lengths(self):
    return [len(v) for v in self.buckets.values()]

  def __len__(self):
    return sum(self.bucket_lengths())

  def is_valid(self):
    return all(
      len(v) >= min_pts for v, min_pts in zip(self.buckets.values(), self.buckets_min_points.values(), strict=True)) \
      and (self.__len__() >= self.min_points_total)

  @staticmethod
  def is_jointly_valid(*pbs: 'PointBuckets'):
    # TODO: this is kind of a hack
    x_bounds = pbs[0].x_bounds
    if not all(pb.x_bounds == x_bounds for pb in pbs):
      raise ValueError("x_bounds must be the same for all PointBuckets")
    return all(
      sum(len(pb.buckets[bounds]) for pb in pbs) >= min_pts
      for bounds, min_pts in zip(x_bounds, pbs[0].buckets_min_points.values(), strict=True)) \
      and (sum(len(pb) for pb in pbs) >= pbs[0].min_points_total)

  def add_point(self, x, y):
    for bounds in self.x_bounds:
      if all((x[i] >= bounds[i][0]) and (x[i] < bounds[i][1]) for i in range(self.x_size)):
        self.buckets[bounds].append([*x, 1.0, y])
        break

  def get_points(self, num_points=None):
    points = np.vstack([x.arr for x in self.buckets.values()])
    if num_points is None:
      return points
    return points[np.random.choice(np.arange(len(points)), min(len(points), num_points), replace=False)]

  def load_points(self, points):
    for x, y in points:
      self.add_point(x, y)
