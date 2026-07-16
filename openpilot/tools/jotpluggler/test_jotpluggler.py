import subprocess
from pathlib import Path


JOTPLUGGLER_DIR = Path(__file__).parent


def test_help():
  result = subprocess.run(["./jotpluggler", "-h"], cwd=JOTPLUGGLER_DIR, capture_output=True, text=True)
  assert result.returncode == 0, result.stderr
  assert "Usage:" in result.stderr
