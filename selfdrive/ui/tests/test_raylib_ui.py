import time
from openpilot.selfdrive.test.helpers import with_processes


@with_processes(["raylib_ui"])
def test_raylib_ui():
  """Test initialization of the UI widgets is successful."""
  time.sleep(1)
