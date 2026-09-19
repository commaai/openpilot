import unittest

import numpy as np

from openpilot.selfdrive.modeld.path_curvature import curvature_from_path


class TestPathCurvature(unittest.TestCase):
  def test_straight(self):
    self.assertEqual(curvature_from_path(np.array([[0, 0], [2, 0], [5, 0]])), 0.0)

  def test_interpolates_by_distance_along_path(self):
    # First leg is 2m; target lies 1m into the next, vertical leg: (2, 1).
    path = np.array([[0, 0], [2, 0], [2, 4]])
    self.assertAlmostEqual(curvature_from_path(path), 2 / 5)

  def test_quarter_circle_both_directions(self):
    angle = np.linspace(0, np.pi / 2, 1001)
    for sign in (-1, 1):
      path = np.column_stack((4 * np.sin(angle), sign * 4 * (1 - np.cos(angle))))
      self.assertAlmostEqual(curvature_from_path(path), sign / 4, places=5)

  def test_duplicate_points(self):
    path = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [4, 0]])
    self.assertEqual(curvature_from_path(path), 0.0)

  def test_short_and_stationary_paths_fall_back(self):
    for path in (np.array([[0, 0], [2.9, 0]]), np.zeros((33, 3))):
      self.assertIsNone(curvature_from_path(path))

  def test_invalid_paths_fall_back(self):
    for path in (np.array([]), np.array([[0, 0]]), np.zeros((3, 1)),
                 np.array([[0, 0], [4, np.nan]]), np.array([[0, 0], [np.inf, 0]])):
      self.assertIsNone(curvature_from_path(path))

  def test_target_behind_car_falls_back(self):
    self.assertIsNone(curvature_from_path(np.array([[0, 0], [-4, 0]])))
