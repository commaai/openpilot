import unittest
from cereal.messaging import SubMaster
from openpilot.common.mock import mock_messages


class TestMock(unittest.TestCase):
  @mock_messages(["liveLocationKalman"])
  def test_liveLocationKalman(self):
    sm = SubMaster(["liveLocationKalman"])
    sm.update(0)
    self.assertTrue(sm.updated["liveLocationKalman"])


if __name__ == "__main__":
  unittest.main()