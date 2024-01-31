import unittest
from cereal.messaging import SubMaster
from openpilot.common.mock import mock_messages


class TestMock(unittest.TestCase):

  @mock_messages(["liveLocationKalman"])
  def test_liveLocationKalman(self):
    sm = SubMaster(["liveLocationKalman"])
    for _ in range(20):
      sm.update()
      self.assertTrue(sm.updated["liveLocationKalman"])

    self.assertTrue(sm.freq_ok["liveLocationKalman"])


if __name__ == "__main__":
  unittest.main()
