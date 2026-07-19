import subprocess
from pathlib import Path

from openpilot.selfdrive.test.helpers import OpenpilotTestCase


JOTPLUGGLER_DIR = Path(__file__).parent


class TestJotpluggler(OpenpilotTestCase):
  def test_help(self):
    result = subprocess.run(["./jotpluggler", "-h"], cwd=JOTPLUGGLER_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stderr
