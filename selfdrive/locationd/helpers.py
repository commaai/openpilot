import numpy as np
import signal
import sys
from typing import List, Optional, Tuple, Any

from cereal import log
from openpilot.common.params import Params
from openpilot.system.swaglog import cloudlog


class NPQueue:
  def __init__(self, maxlen: int, rowsize: int) -> None:
    self.maxlen = maxlen
    self.arr = np.empty((0, rowsize))

  def __len__(self) -> int:
    return len(self.arr)

  def append(self, pt: List[float]) -> None:
    if len(self.arr) < self.maxlen:
      self.arr = np.append(self.arr, [pt], axis=0)
    else:
      self.arr[:-1] = self.arr[1:]
      self.arr[-1] = pt


class PointBuckets:
  def __init__(self, x_bounds: List[Tuple[float, float]], min_points: List[float], min_points_total: int, points_per_bucket: int, rowsize: int) -> None:
    self.x_bounds = x_bounds
    self.buckets = {bounds: NPQueue(maxlen=points_per_bucket, rowsize=rowsize) for bounds in x_bounds}
    self.buckets_min_points = dict(zip(x_bounds, min_points, strict=True))
    self.min_points_total = min_points_total

  def bucket_lengths(self) -> List[int]:
    return [len(v) for v in self.buckets.values()]

  def __len__(self) -> int:
    return sum(self.bucket_lengths())

  def is_valid(self) -> bool:
    individual_buckets_valid = all(len(v) >= min_pts for v, min_pts in zip(self.buckets.values(), self.buckets_min_points.values(), strict=True))
    total_points_valid = self.__len__() >= self.min_points_total
    return individual_buckets_valid and total_points_valid

  def add_point(self, x: float, y: float, bucket_val: float) -> None:
    raise NotImplementedError

  def get_points(self, num_points: Optional[int] = None) -> Any:
    points = np.vstack([x.arr for x in self.buckets.values()])
    if num_points is None:
      return points
    return points[np.random.choice(np.arange(len(points)), min(len(points), num_points), replace=False)]

  def load_points(self, points: List[List[float]]) -> None:
    for point in points:
      self.add_point(*point)


class ParameterEstimator:
  """ Base class for parameter estimators """
  def reset(self) -> None:
    raise NotImplementedError

  def handle_log(self, t: int, which: str, msg: log.Event) -> None:
    raise NotImplementedError

  def get_msg(self, valid: bool, with_points: bool) -> log.Event:
    raise NotImplementedError


def cache_points_onexit(param_name, estimator, sig, frame):
  signal.signal(sig, signal.SIG_DFL)
  cloudlog.warning(f"Caching {param_name} param")
  params = Params()
  msg = estimator.get_msg(valid=True, with_points=True)
  params.put(param_name, msg.to_bytes())
  sys.exit(0)
