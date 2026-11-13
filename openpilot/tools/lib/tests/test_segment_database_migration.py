import unittest
from unittest.mock import Mock, patch

import requests

from openpilot.tools.lib.comma_car_segments import get_comma_car_segments_database


class TestSegmentDatabaseMigration(unittest.TestCase):
  def test_migrated_platforms_keep_all_segments_and_remove_only_the_selector(self):
    response = Mock()
    response.json.return_value = {"OLD": ["device/stops", "device/suffix/s"], "NEW": ["device/third/s"]}
    with patch("opendbc.car.fingerprints.MIGRATION", {"OLD": "NEW"}), patch("requests.get", return_value=response):
      self.assertEqual(get_comma_car_segments_database(), {"NEW": ["device/stops", "device/suffix", "device/third"]})

  def test_database_http_failure_is_reported_before_decoding(self):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("503")
    with patch("requests.get", return_value=response), self.assertRaises(requests.HTTPError):
      get_comma_car_segments_database()
    response.json.assert_not_called()
