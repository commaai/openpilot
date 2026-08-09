import math
import unittest

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.hardware.hardwared import update_filtered_temperature


class TestTemperatureFiltering(unittest.TestCase):
  def test_ignores_invalid_readings(self):
    temp_filter = FirstOrderFilter(0.0, 5.0, 0.5, initialized=False)

    temperature, valid = update_filtered_temperature(temp_filter, [math.nan, 42.5, math.inf])

    self.assertTrue(valid)
    self.assertEqual(temperature, 42.5)

  def test_all_invalid_readings_do_not_poison_filter(self):
    temp_filter = FirstOrderFilter(42.5, 5.0, 0.5)

    temperature, valid = update_filtered_temperature(temp_filter, [math.nan, math.inf])

    self.assertFalse(valid)
    self.assertTrue(math.isinf(temperature))
    self.assertEqual(temp_filter.x, 42.5)
