import unittest
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.ui.mici.onroad.stop_bar import HeldStopBar


def prediction(speeds=(8, 5, 2, 1, 1), acceleration=-0.5, x=(0, 7, 12, 14, 15)):
  return SimpleNamespace(
    velocity=SimpleNamespace(t=[0, 1, 2, 3, 4], x=list(speeds)),
    position=SimpleNamespace(t=[0, 1, 2, 3, 4], x=list(x), y=[0] * 5, z=[0] * 5),
    action=SimpleNamespace(desiredAcceleration=acceleration),
  )


class TestHeldStopBar(unittest.TestCase):
  def setUp(self):
    self.bar = HeldStopBar()

  def update(self, model=None, speed=8, yaw=0, time=0, model_time=None, enabled=True):
    self.bar.update(model or prediction(), speed, yaw, time, time if model_time is None else model_time, enabled)

  def test_rolling_stop_seeds_and_later_creep_does_not_reanchor(self):
    self.update()
    np.testing.assert_allclose(self.bar.point, [14, 0, 0])
    # A subsequent forward-looking creep prediction is not a new target.
    self.update(prediction((1, 1, 1, 1, 1), 0.05, (0, 20, 30, 40, 50)), speed=1, time=0.1)
    np.testing.assert_allclose(self.bar.point, [14 - 0.45, 0, 0])
    # Advance on car messages, even while rendering the same model frame.
    self.update(speed=1, time=0.2, model_time=0.1)
    np.testing.assert_allclose(self.bar.point, [14 - 0.55, 0, 0])
    previous = self.bar.point.copy()
    self.update(speed=1, time=0.2, model_time=0.1)
    np.testing.assert_array_equal(self.bar.point, previous)

  def test_no_seed_for_cruising_or_small_slowdown(self):
    self.update(prediction((8, 8, 8, 8, 8)))
    self.assertIsNone(self.bar.point)
    self.update(prediction((2, 1.8, 1.5, 1.5, 1.5)), speed=2, time=0.1)
    self.assertIsNone(self.bar.point)

  def test_move_off_requires_consistency_and_does_not_use_render_count(self):
    self.update()
    moving = prediction((1, 3, 5, 6, 7), 0.5)
    self.update(moving, speed=1, time=0.1)
    self.update(moving, speed=1, time=0.4, model_time=0.1)
    self.assertIsNotNone(self.bar.point)
    self.update(prediction((1, 1, 1, 1, 1), 0.05), speed=1, time=0.5)
    self.assertIsNone(self.bar.move_off_since)
    for time in (0.6, 0.8, 1.0):
      self.update(moving, speed=1, time=time)
      self.assertIsNotNone(self.bar.point)
    self.update(moving, speed=1, time=1.2)
    self.assertIsNone(self.bar.point)

  def test_passed_target_and_disengagement_clear(self):
    self.update()
    self.bar.point[0] = 0.5
    self.update(speed=8, time=0.1)
    self.assertIsNone(self.bar.point)
    self.update(time=0.2)
    self.assertIsNotNone(self.bar.point)
    self.update(time=0.3, enabled=False)
    self.assertIsNone(self.bar.point)

  def test_odometry_turn_direction(self):
    self.update(speed=8, yaw=0.2)
    initial = self.bar.point.copy()
    self.update(speed=8, yaw=0.2, time=0.1, model_time=0)
    # Turning left moves an initially straight-ahead road point to our right.
    self.assertGreater(self.bar.point[1], initial[1])
    self.assertLess(self.bar.point[0], initial[0])
    self.assertAlmostEqual(self.bar.heading, 0.02)

  def test_stale_or_invalid_state_resets(self):
    self.update()
    cruising = prediction((8, 8, 8, 8, 8))
    self.update(cruising, time=3)
    self.assertIsNone(self.bar.point)
    self.update(time=3.1)
    self.assertIsNotNone(self.bar.point)
    self.update(cruising, time=1)
    self.assertIsNone(self.bar.point)
    self.update(time=1.1)
    invalid = prediction()
    invalid.velocity.x[2] = float('nan')
    self.update(invalid, time=1.2)
    self.assertIsNone(self.bar.point)

  def test_projection_height_and_width_fallback(self):
    transform = np.array([[240, 500, 0], [100, 0, 500], [1, 0, 0]], dtype=np.float32)
    empty = np.empty((0, 3))
    for distance in (6, 10, 20, 40, 70, 100):
      self.bar.point = np.array([distance, 0, 0], dtype=float)
      self.bar.project(transform, 1.2, [empty] * 4, [0] * 4, [empty] * 2, [1] * 2)
      points = self.bar.points
      self.assertEqual(points.shape, (4, 2))
      self.assertGreaterEqual(np.ptp(points[:, 1]), 6 - 1e-4)
      self.assertLessEqual(np.ptp(points[:, 1]), 12 + 1e-4)
      x = 600 / (points[:, 1] - 100)
      y = (points[:, 0] - 240) * x / 500
      np.testing.assert_allclose(y, [-1.8, 1.8, 1.8, -1.8], atol=1e-4)

  def test_projection_fits_green_boundaries(self):
    x = np.linspace(0, 100, 101)
    left = np.column_stack((x, np.full_like(x, -2), np.full_like(x, 1.2)))
    right = np.column_stack((x, np.full_like(x, 2.2), np.full_like(x, 1.2)))
    empty = np.empty((0, 3))
    transform = np.array([[240, 500, 0], [100, 0, 500], [1, 0, 0]], dtype=np.float32)
    self.bar.point = np.array([20, 0, 0], dtype=float)
    for probs, stds in (([0, 0.9, 0.9, 0], [1, 1]), ([0, 0, 0, 0], [0.1, 0.1])):
      self.bar.project(transform, 1.2, [empty, left, right, empty], probs, [left, right], stds)
      points = self.bar.points
      px = 600 / (points[:, 1] - 100)
      py = (points[:, 0] - 240) * px / 500
      np.testing.assert_allclose(py, [-2, 2.2, 2.2, -2], atol=1e-4)


if __name__ == '__main__':
  unittest.main()
