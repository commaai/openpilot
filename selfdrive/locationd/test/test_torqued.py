from cereal import car
from openpilot.selfdrive.locationd.torqued import TorqueEstimator


def test_cal_percent():
  est = TorqueEstimator(car.CarParams())
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 0

  for (low, high), min_pts in zip(est.filtered_points.buckets.keys(),
                                  est.filtered_points.buckets_min_points.values(), strict=True):
    print(f"Testing bucket {low} to {high} with min_pts {min_pts}")
    for _ in range(int(min_pts)):
      est.filtered_points.add_point((low + high) / 2.0, 0.0)
  print(est.filtered_points.is_calculable(), est.filtered_points.is_valid(), est.filtered_points.get_valid_percent(), len(est.filtered_points))

  # enough bucket points, but not enough total points
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == (len(est.filtered_points) / est.min_points_total * 100 + 100) / 2

  # add enough points to bucket with most capacity
  key = list(est.filtered_points.buckets)[0]
  for _ in range(est.min_points_total - len(est.filtered_points) - 1):
    est.filtered_points.add_point((key[0] + key[1]) / 2.0, 0.0)

  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 99

  est.filtered_points.add_point((key[0] + key[1]) / 2.0, 0.0)
  msg = est.get_msg()
  assert msg.liveTorqueParameters.calPerc == 100
