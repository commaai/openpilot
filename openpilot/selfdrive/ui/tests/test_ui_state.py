from types import SimpleNamespace

from openpilot.selfdrive.ui.ui_state import ChestnutState, UIState


class SubMaster:
  def __init__(self):
    self.recv_frame = {"modelV2": 0}
    self.alive = {"modelV2": False}
    self.data = {
      "deviceState": SimpleNamespace(chestnutPresent=True),
      "modelV2": SimpleNamespace(big=False),
    }

  def __getitem__(self, name):
    return self.data[name]


def ui_state():
  ui = UIState.__new__(UIState)
  ui.sm = SubMaster()
  ui.started = True
  ui.started_frame = 0
  ui.chestnut_present = True
  ui.chestnut_compiled = True
  ui.chestnut_loading = False
  ui.chestnut_model_error = False
  ui.chestnut_state = ChestnutState.READY
  return ui


def test_model_is_loading_before_first_output():
  ui = ui_state()
  ui._update_chestnut_state()
  assert ui.chestnut_state == ChestnutState.LOADING


def test_model_failure_is_sticky():
  ui = ui_state()
  ui.chestnut_model_error = True
  ui._update_chestnut_state()
  assert ui.chestnut_state == ChestnutState.FAILED


def test_small_model_output_is_failed():
  ui = ui_state()
  ui._update_chestnut_state()
  assert ui.chestnut_state == ChestnutState.LOADING

  ui.sm.recv_frame["modelV2"] = 1
  ui.sm.alive["modelV2"] = True
  ui._update_chestnut_state()
  assert ui.chestnut_state == ChestnutState.FAILED


def test_successful_model_attempt_becomes_active():
  ui = ui_state()
  ui.chestnut_loading = True
  ui._update_chestnut_state()

  ui.chestnut_loading = False
  ui.sm.recv_frame["modelV2"] = 1
  ui.sm.alive["modelV2"] = True
  ui.sm.data["modelV2"].big = True
  ui._update_chestnut_state()
  assert ui.chestnut_state == ChestnutState.ACTIVE
