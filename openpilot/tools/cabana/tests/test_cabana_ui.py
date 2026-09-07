import subprocess
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase

CABANA_DIR = Path(__file__).parent.parent


class TestCabanaUi(OpenpilotTestCase):
  def test_help(self):
    result = subprocess.run(["./cabana", "-h"], cwd=CABANA_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stderr

  def test_layout_optional_value(self):
    for args in (["--layout"], ["--layout="], ["--layout", "layouts/tuning.json"], ["--layout=layouts/tuning.json"]):
      with self.subTest(args=args):
        result = subprocess.run(["./cabana", *args, "--help"], cwd=CABANA_DIR, capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        assert "Usage:" in result.stderr

  def test_layout_does_not_consume_options(self):
    result = subprocess.run(["./cabana", "--layout", "--unknown-option", "--help"],
                            cwd=CABANA_DIR, capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "unknown option --unknown-option" in result.stderr
