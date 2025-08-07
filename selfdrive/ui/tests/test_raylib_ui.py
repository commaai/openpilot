from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout


def test_ui():
  """Test initialization of the UI widgets is successful."""
  gui_app.init_window("UI")
  MainLayout()
