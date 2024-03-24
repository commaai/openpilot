import pytest
import unittest
import requests
from openpilot.selfdrive.car.fingerprints import MIGRATION
from openpilot.tools.lib.comma_car_segments import get_comma_car_segments_database, get_url
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import SegmentRange


@pytest.mark.skip(reason="huggingface is flaky, run this test manually to check for issues")
class TestCommaCarSegments(unittest.TestCase):
  def test_database(self):
    database = get_comma_car_segments_database()

    platforms = database.keys()

    assert len(platforms) > 100

  def test_download_segment(self):
    database = get_comma_car_segments_database()

    fp = "SUBARU_FORESTER"

    segment = database[fp][0]

    sr = SegmentRange(segment)

    url = get_url(sr.route_name, sr.slice)

    resp = requests.get(url)
    self.assertEqual(resp.status_code, 200)

    lr = LogReader(url)
    CP = lr.first("carParams")
    self.assertEqual(MIGRATION.get(CP.carFingerprint, CP.carFingerprint), fp)


if __name__ == "__main__":
  unittest.main()
