#!/usr/bin/env python3
# pylint: skip-file

import unittest

import numpy as np

from common.transformations.tests.test_coordinates import (
    ecef_init_batch, ecef_positions, ecef_positions_offset,
    ecef_positions_offset_batch, geodetic_positions, ned_offsets,
    ned_offsets_batch)
from common.transformations.transformations import (LocalCoord,
                                                    ecef2geodetic_single,
                                                    geodetic2ecef_single)


class TestCoordinatesCython(unittest.TestCase):
  def test_geodetic2ecef(self):
    for i in range(len(geodetic_positions)):
      np.testing.assert_allclose(ecef_positions[i], geodetic2ecef_single(geodetic_positions[i]), rtol=1e-9)

  def test_ecef2geodetic(self):
    for i in range(len(geodetic_positions)):
      # Higher tolerance on altitude
      np.testing.assert_allclose(geodetic_positions[i][: 2], ecef2geodetic_single(ecef_positions[i])[: 2], rtol=1e-9)
      np.testing.assert_allclose(geodetic_positions[i][2], ecef2geodetic_single(ecef_positions[i])[2], rtol=1e-9, atol=1e-4)

  def test_small_distances(self):
    start_geodetic = np.array([33.8042184, -117.888593, 0.0])
    local_coord = LocalCoord.from_geodetic(start_geodetic)

    start_ned = local_coord.geodetic2ned_single(start_geodetic)
    np.testing.assert_array_equal(start_ned, np.zeros(3,))

    west_geodetic = start_geodetic + [0, -0.0005, 0]
    west_ned = local_coord.geodetic2ned_single(west_geodetic)
    self.assertLess(np.abs(west_ned[0]), 1e-3)
    self.assertLess(west_ned[1], 0)

    southwest_geodetic = start_geodetic + [-0.0005, -0.002, 0]
    southwest_ned = local_coord.geodetic2ned_single(southwest_geodetic)
    self.assertLess(southwest_ned[0], 0)
    self.assertLess(southwest_ned[1], 0)

  def test_ned(self):
    for ecef_pos in ecef_positions:
      converter = LocalCoord.from_ecef(ecef_pos)
      ecef_pos_moved = ecef_pos + [25, -25, 25]
      ecef_pos_moved_double_converted = converter.ned2ecef_single(converter.ecef2ned_single(ecef_pos_moved))
      np.testing.assert_allclose(ecef_pos_moved, ecef_pos_moved_double_converted, rtol=1e-9)

    for geo_pos in geodetic_positions:
      converter = LocalCoord.from_geodetic(geo_pos)
      geo_pos_moved = geo_pos + np.array([0, 0, 10])
      geo_pos_double_converted_moved = converter.ned2geodetic_single(converter.geodetic2ned_single(geo_pos) + np.array([0, 0, -10]))
      np.testing.assert_allclose(geo_pos_moved[:2], geo_pos_double_converted_moved[:2], rtol=1e-9, atol=1e-6)
      np.testing.assert_allclose(geo_pos_moved[2], geo_pos_double_converted_moved[2], rtol=1e-9, atol=1e-4)

  def test_ned_saved_results(self):
    for i, ecef_pos in enumerate(ecef_positions):
      converter = LocalCoord.from_ecef(ecef_pos)
      np.testing.assert_allclose(converter.ned2ecef_single(ned_offsets[i]),
                                 ecef_positions_offset[i],
                                 rtol=1e-9, atol=1e-4)
      np.testing.assert_allclose(converter.ecef2ned_single(ecef_positions_offset[i]),
                                 ned_offsets[i],
                                 rtol=1e-9, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
