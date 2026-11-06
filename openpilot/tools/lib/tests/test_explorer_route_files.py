import tempfile
import unittest
from pathlib import Path

from openpilot.tools.lib.route import Route


class TestExplorerRouteFiles(unittest.TestCase):
  def test_underscore_segment_files(self):
    name = "a2a0ccea32023010|2023-07-27--13-01-19"
    with tempfile.TemporaryDirectory() as tmp:
      prefix = name.replace("|", "_") + "--0"
      log = Path(tmp, prefix + "--qlog.zst")
      log.touch()
      camera = Path(tmp, prefix, "fcamera.hevc")
      camera.parent.mkdir()
      camera.touch()

      route = Route(name, data_dir=tmp)
      self.assertEqual(route.qlog_paths(), [str(log)])
      self.assertEqual(route.camera_paths(), [str(camera)])
