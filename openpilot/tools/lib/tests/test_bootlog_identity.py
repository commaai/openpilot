import unittest

from openpilot.tools.lib.bootlog import Bootlog


class TestBootlogIdentity(unittest.TestCase):
  def test_equal_timestamps_on_different_devices_are_distinct(self):
    path = "1111111111111111/2020-01-01--00-00-00"
    first = Bootlog(path)
    other = Bootlog(path.replace("1111111111111111", "2222222222222222"))
    self.assertEqual(first, Bootlog("https://example.com/" + path))
    self.assertNotEqual(first, other)
    self.assertLess(first, other)
    self.assertFalse(other < first)
    self.assertLess(first, Bootlog("0000000000000000/2020-01-02--00-00-00"))
