import subprocess
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase

CABANA_DIR = Path(__file__).parent.parent


class TestCabanaUi(OpenpilotTestCase):
  def test_help(self):
    result = subprocess.run(["./_cabana_ui", "-h"], cwd=CABANA_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stderr
