import numpy as np

from cereal import car
from openpilot.selfdrive.locationd.torqued import TorqueEstimator


def test_calPerc_progress():
  est = TorqueEstimator(car.CarParams(), decimated=True)
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 0

  for (low, high), req in zip(est.filtered_points.buckets.keys(),
                              est.filtered_points.buckets_min_points.values()):
    for _ in range(int(req)):
      est.filtered_points.add_point((low + high) / 2.0, 0.0)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 100
