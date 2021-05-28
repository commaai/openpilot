import unittest
from decimal import Decimal
from math import pi
import numpy as np


class TestMapsdGeoLibrary(unittest.TestCase):
  # 0. coord to rad point tuple conversion
  def test_coord_to_rad(self):
    points = [
        (0., 360.),
        (Decimal(0.), Decimal(360.)),
        (Decimal(180.), Decimal(540.)),
    ]
    expected = [
        (0., 2 * pi),
        (0., 2 * pi),
        (pi, 3 * pi),
    ]
    rad_tuples = list(map(lambda p: np.radians(p), points))
    self.assertEqual(rad_tuples, expected)

  # Helpers
  def _assertAlmostEqualList(self, a, b, places=0):
      for idx, el_a in enumerate(a):
        self.assertAlmostEqual(el_a, b[idx], places)

  def _assertAlmostEqualListOfTuples(self, a, b, places=0):
    for idx, el_a in enumerate(a):
      for idy, el_el_a in enumerate(el_a):
        self.assertAlmostEqual(el_el_a, b[idx][idy], places)
