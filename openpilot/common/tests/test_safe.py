import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.common import safe


class TestSafe(unittest.TestCase):
  def test_read_file(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      path = Path(tmp_dir) / "file"
      path.write_bytes(b"contents")

      self.assertEqual(safe.read_file(path, b"default"), b"contents")
      self.assertEqual(safe.read_file(Path(tmp_dir) / "missing", b"default"), b"default")

  def test_check_output_default(self):
    for exception in (OSError(), subprocess.CalledProcessError(1, ["command"])):
      with self.subTest(exception=exception), patch("subprocess.check_output", side_effect=exception):
        self.assertEqual(safe.check_output(["command"], b"default"), b"default")
