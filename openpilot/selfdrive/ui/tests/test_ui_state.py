from openpilot.selfdrive.ui.ui_state import Device


def test_interactive_timeout_callbacks_are_unique_and_removable():
  device = Device()

  def callback():
    pass

  device.add_interactive_timeout_callback(callback)
  device.add_interactive_timeout_callback(callback)
  assert device._interactive_timeout_callbacks == [callback]

  device.remove_interactive_timeout_callback(callback)
  device.remove_interactive_timeout_callback(callback)
  assert device._interactive_timeout_callbacks == []


def test_interactive_timeout_callback_can_remove_itself(monkeypatch):
  device = Device()
  monkeypatch.setattr(device, "_set_awake", lambda on: None)

  calls = []

  def first_callback():
    calls.append("first")
    device.remove_interactive_timeout_callback(first_callback)

  def second_callback():
    calls.append("second")

  device.add_interactive_timeout_callback(first_callback)
  device.add_interactive_timeout_callback(second_callback)

  device._interaction_time = 0
  device._prev_timed_out = False
  device._update_wakefulness()

  assert calls == ["first", "second"]
  assert device._interactive_timeout_callbacks == [second_callback]
