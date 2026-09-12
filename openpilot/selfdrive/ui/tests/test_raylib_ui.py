import time
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.helpers import with_processes


class TestRaylibUi(OpenpilotTestCase):
  @with_processes(["ui"])
  def test_raylib_ui(self):
    """Test initialization of the UI widgets is successful."""
    time.sleep(1)
