import pyray as rl
rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)

from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.ui_state import ui_state, device


def test_ui_state_callbacks():
  gui_app.init_window("test-ui")

  from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
  layout = MiciMainLayout()

  for cb in ui_state._offroad_transition_callbacks:
    cb()

  for cb in ui_state._engaged_transition_callbacks:
    cb()

  for cb in device._interactive_timeout_callbacks:
    cb()

  del layout
  gui_app.close()


if __name__ == "__main__":
  test_ui_state_callbacks()
