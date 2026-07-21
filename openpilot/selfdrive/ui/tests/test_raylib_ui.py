import time
from openpilot.selfdrive.test.helpers import with_processes


class TestRaylibUi:
  @with_processes(["ui"])
  def test_raylib_ui(self):
    """Test initialization of the UI widgets is successful."""
    time.sleep(1)
