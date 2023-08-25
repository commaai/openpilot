#!/usr/bin/env python3

import numpy as np
import unittest

import openpilot.common.transformations.coordinates as coord

geodetic_positions = np.array([[37.7610403, -122.4778699, 115],
                                 [27.4840915, -68.5867592, 2380],
                                 [32.4916858, -113.652821, -6],
                                 [15.1392514, 103.6976037, 24],
                                 [24.2302229, 44.2835412, 1650]])

ecef_positions = np.array([[-2711076.55270557, -4259167.14692758,  3884579.87669935],
                          [ 2068042.69652729, -5273435.40316622,  2927004.89190746],
                          [-2160412.60461669, -4932588.89873832,  3406542.29652851],
                          [-1458247.92550567,  5983060.87496612,  1654984.6099885 ],
                          [ 4167239.10867871,  4064301.90363223,  2602234.6065749 ]])

ecef_positions_offset = np.array([[-2711004.46961115, -4259099.33540613,  3884605.16002147],
                                  [ 2068074.30639499, -5273413.78835412,  2927012.48741131],
                                  [-2160344.53748176, -4932586.20092211,  3406636.2962545 ],
                                  [-1458211.98517094,  5983151.11161276,  1655077.02698447],
                                  [ 4167271.20055269,  4064398.22619263,  2602238.95265847]])


ned_offsets = np.array([[78.722153649976391, 24.396208657446344, 60.343017506838436],
                       [10.699003365155221, 37.319278617604269, 4.1084100025050407],
                       [95.282646251726959, 61.266689955574428, -25.376506058505054],
                       [68.535769283630003, -56.285970011848889, -100.54840137956515],
                       [-33.066609321880179, 46.549821994306861, -84.062540548335591]])

ecef_init_batch = np.array([2068042.69652729, -5273435.40316622,  2927004.89190746])
ecef_positions_offset_batch = np.array([[ 2068089.41454771, -5273434.46829148,  2927074.04783672],
                                        [ 2068103.31628647, -5273393.92275431,  2927102.08725987],
                                        [ 2068108.49939636, -5273359.27047121,  2927045.07091581],
                                        [ 2068075.12395611, -5273381.69432566,  2927041.08207992],
                                        [ 2068060.72033399, -5273430.6061505,  2927094.54928305]])

ned_offsets_batch = np.array([[  53.88103168,   43.83445935,  -46.27488057],
                              [  93.83378995,   71.57943024,  -30.23113187],
                              [  57.26725796,   89.05602684,   23.02265814],
                              [  49.71775195,   49.79767572,   17.15351015],
                              [  78.56272609,   18.53100158,  -43.25290759]])


class TestNED(unittest.TestCase):
  def test_small_distances(self):
    start_geodetic = np.array([33.8042184, -117.888593, 0.0])
    local_coord = coord.LocalCoord.from_geodetic(start_geodetic)

    start_ned = local_coord.geodetic2ned(start_geodetic)
    np.testing.assert_array_equal(start_ned, np.zeros(3,))

    west_geodetic = start_geodetic + [0, -0.0005, 0]
    west_ned = local_coord.geodetic2ned(west_geodetic)
    self.assertLess(np.abs(west_ned[0]), 1e-3)
    self.assertLess(west_ned[1], 0)

    southwest_geodetic = start_geodetic + [-0.0005, -0.002, 0]
    southwest_ned = local_coord.geodetic2ned(southwest_geodetic)
    self.assertLess(southwest_ned[0], 0)
    self.assertLess(southwest_ned[1], 0)

  def test_ecef_geodetic(self):
    # testing single
    np.testing.assert_allclose(ecef_positions[0], coord.geodetic2ecef(geodetic_positions[0]), rtol=1e-9)
    np.testing.assert_allclose(geodetic_positions[0, :2], coord.ecef2geodetic(ecef_positions[0])[:2], rtol=1e-9)
    np.testing.assert_allclose(geodetic_positions[0, 2], coord.ecef2geodetic(ecef_positions[0])[2], rtol=1e-9, atol=1e-4)

    np.testing.assert_allclose(geodetic_positions[:, :2], coord.ecef2geodetic(ecef_positions)[:, :2], rtol=1e-9)
    np.testing.assert_allclose(geodetic_positions[:, 2], coord.ecef2geodetic(ecef_positions)[:, 2], rtol=1e-9, atol=1e-4)
    np.testing.assert_allclose(ecef_positions, coord.geodetic2ecef(geodetic_positions), rtol=1e-9)


  def test_ned(self):
    for ecef_pos in ecef_positions:
      converter = coord.LocalCoord.from_ecef(ecef_pos)
      ecef_pos_moved = ecef_pos + [25, -25, 25]
      ecef_pos_moved_double_converted = converter.ned2ecef(converter.ecef2ned(ecef_pos_moved))
      np.testing.assert_allclose(ecef_pos_moved, ecef_pos_moved_double_converted, rtol=1e-9)

    for geo_pos in geodetic_positions:
      converter = coord.LocalCoord.from_geodetic(geo_pos)
      geo_pos_moved = geo_pos + np.array([0, 0, 10])
      geo_pos_double_converted_moved = converter.ned2geodetic(converter.geodetic2ned(geo_pos) + np.array([0, 0, -10]))
      np.testing.assert_allclose(geo_pos_moved[:2], geo_pos_double_converted_moved[:2], rtol=1e-9, atol=1e-6)
      np.testing.assert_allclose(geo_pos_moved[2], geo_pos_double_converted_moved[2], rtol=1e-9, atol=1e-4)

  def test_ned_saved_results(self):
    for i, ecef_pos in enumerate(ecef_positions):
      converter = coord.LocalCoord.from_ecef(ecef_pos)
      np.testing.assert_allclose(converter.ned2ecef(ned_offsets[i]),
                                 ecef_positions_offset[i],
                                 rtol=1e-9, atol=1e-4)
      np.testing.assert_allclose(converter.ecef2ned(ecef_positions_offset[i]),
                                 ned_offsets[i],
                                 rtol=1e-9, atol=1e-4)

  def test_ned_batch(self):
    converter = coord.LocalCoord.from_ecef(ecef_init_batch)
    np.testing.assert_allclose(converter.ecef2ned(ecef_positions_offset_batch),
                                                           ned_offsets_batch,
                                                           rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(converter.ned2ecef(ned_offsets_batch),
                                                           ecef_positions_offset_batch,
                                                           rtol=1e-9, atol=1e-7)
if __name__ == "__main__":
  unittest.main()
