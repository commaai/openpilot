import unittest

from openpilot.tools.lib.srreader import SegmentRangeReader, openpilotci_source


class TestSegmentRangeReader(unittest.TestCase):
  def test_logreader(self):
    sr = SegmentRangeReader("1bbe6bf2d62f58a8|2022-07-14--17-11-43--10", source=openpilotci_source)

    for m in sr:
      print(m.which())

if __name__ == "__main__":
  unittest.main()
