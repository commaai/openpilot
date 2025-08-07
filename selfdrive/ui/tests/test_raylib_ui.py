import time
from openpilot.selfdrive.test.helpers import with_processes


@with_processes(["raylib_ui"])
def test_raylib_ui():
  time.sleep(1)
