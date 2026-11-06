import unittest
from unittest.mock import patch

from openpilot.tools.lib.filereader import internal_source_available


class TestInternalSourcePort(unittest.TestCase):
  def test_probes_endpoint_port(self):
    internal_source_available.cache_clear()
    self.addCleanup(internal_source_available.cache_clear)
    for scheme, authority, port in (("https", "logs.test", 443), ("http", "logs.test", 80), ("https", "logs.test:9443", 9443)):
      with self.subTest(scheme=scheme, port=port), patch("openpilot.tools.lib.filereader.socket.socket") as sock:
        self.assertTrue(internal_source_available(f"{scheme}://{authority}/data/"))
        sock.return_value.__enter__.return_value.connect.assert_called_once_with(("logs.test", port))
