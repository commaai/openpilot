import time
from openpilot.selfdrive.test.helpers import OpenpilotTestCase, with_processes


class TestRaylibUi(OpenpilotTestCase):
  @with_processes(["ui"])
  def test_raylib_ui(self):
    """Test initialization of the UI widgets is successful."""
    time.sleep(1)
