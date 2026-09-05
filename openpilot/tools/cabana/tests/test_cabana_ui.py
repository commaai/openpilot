import subprocess
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase

CABANA_DIR = Path(__file__).parent.parent


class TestCabanaUi(OpenpilotTestCase):
  def test_help(self):
    result = subprocess.run(["./_cabana_ui", "-h"], cwd=CABANA_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stderr
    assert "--log-layout" in result.stderr

  def test_log_arguments(self):
    for args in (["--log"], ["--log", "--msgq"], ["--log-layout"]):
      result = subprocess.run(["./_cabana_ui", *args], cwd=CABANA_DIR, capture_output=True, text=True, timeout=10)
      assert result.returncode == 1, result.stderr
      assert "error:" in result.stderr

  def test_log_signals(self):
    result = subprocess.run(["./tests/test_logsignals"], cwd=CABANA_DIR, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr

  def test_log_replay(self):
    result = subprocess.run(["./tests/test_log_replay"], cwd=CABANA_DIR, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr

  def test_log_panel(self):
    result = subprocess.run(["./tests/test_logpanel"], cwd=CABANA_DIR, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
