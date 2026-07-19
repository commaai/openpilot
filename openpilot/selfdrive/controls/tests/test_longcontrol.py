from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, long_control_state_trans


class TestLongControlStateTransition:
  def test_active(self):
    assert long_control_state_trans(active=True) == LongCtrlState.pid

  def test_inactive(self):
    assert long_control_state_trans(active=False) == LongCtrlState.off
