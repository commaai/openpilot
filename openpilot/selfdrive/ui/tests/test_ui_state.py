import unittest
import sys
import types
from unittest import mock

from openpilot.cereal import messaging

application_module = types.ModuleType("openpilot.system.ui.lib.application")
application_module.gui_app = mock.Mock(target_fps=20)
with mock.patch.dict(sys.modules, {"openpilot.system.ui.lib.application": application_module}), \
     mock.patch.object(messaging, "SubMaster"):
  from openpilot.selfdrive.ui.ui_state import Device


class TestInteractiveTimeoutCallbacks(unittest.TestCase):
  def test_add_remove_callback(self):
    device = Device()

    def callback():
      pass

    device.add_interactive_timeout_callback(callback)
    device.add_interactive_timeout_callback(callback)
    assert device._interactive_timeout_callbacks == [callback]

    device.remove_interactive_timeout_callback(callback)
    device.remove_interactive_timeout_callback(callback)
    assert device._interactive_timeout_callbacks == []

  def test_callback_can_remove_itself(self):
    device = Device()
    calls = []

    def first_callback():
      calls.append("first")
      device.remove_interactive_timeout_callback(first_callback)

    def second_callback():
      calls.append("second")

    device.add_interactive_timeout_callback(first_callback)
    device.add_interactive_timeout_callback(second_callback)
    device._run_interactive_timeout_callbacks()

    assert calls == ["first", "second"]
    assert device._interactive_timeout_callbacks == [second_callback]
