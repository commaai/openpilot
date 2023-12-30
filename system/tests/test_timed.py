#!/usr/bin/env python3
import unittest

from openpilot.system.timed import get_modem_time_output, calculate_time_zone_offset, determine_time_zone, parse_and_format_utc_date


class TestTimed(unittest.TestCase):

  def test_get_modem_time_output_length(self):
    output = get_modem_time_output()
    self.assertEqual(len(output), 46)


  def test_calculate_time_zone_offset(self):
    known_output = "response: '+QLTS: \"2023/12/30,02:51:17-20,0\"' "
    expected_offset = -20
    self.assertEqual(calculate_time_zone_offset(known_output), expected_offset)


  def test_determine_time_zone(self):
    test_cases = [(-4, 'Etc/GMT+1'), (4, 'Etc/GMT-1'), (0, 'Etc/GMT+0')]
    for offset, expected_timezone in test_cases:
      self.assertEqual(determine_time_zone(offset), expected_timezone)


  def test_parse_and_format_utc_date(self):
    known_output = "response: '+QLTS: \"2023/12/30,02:51:17-20,0\"' "
    expected_utc_date = "2023-12-30 02:51:17"
    self.assertEqual(parse_and_format_utc_date(known_output), expected_utc_date)


if __name__ == '__main__':
  unittest.main()
