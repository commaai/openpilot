import unittest
from unittest.mock import patch
from urllib3.response import HTTPResponse

from openpilot.tools.lib.url_file import URLFile


class TestIgnoredRange(unittest.TestCase):
  def test_seek_and_disjoint_ranges(self):
    f = URLFile("https://example.com/log", cache=False)
    with patch.object(f, "_request", return_value=HTTPResponse(body=b"0123456789", status=200)):
      f.seek(4)
      self.assertEqual(f.read(3), b"456")
      self.assertEqual(f.tell(), 7)
      self.assertEqual(f.get_multi_range([(6, 10), (1, 3)]), [b"6789", b"12"])
