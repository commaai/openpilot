from __future__ import annotations

from openpilot.selfdrive.ui.ui_state import device
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, GreyBigButton
from openpilot.selfdrive.ui.tss3_oracle_runner import STAGE_LABELS, Tss3OracleRunner
from openpilot.system.ui.widgets.scroller import NavScroller

_oracle_bringup_active = False


def oracle_bringup_active() -> bool:
  return _oracle_bringup_active


class Tss3OracleBringupPage(NavScroller):
  """Native comma-four page for the TSS3 RAM-oracle startup flow."""

  def __init__(self):
    super().__init__()
    self._runner = Tss3OracleRunner()

    self._status_card = GreyBigButton("oracle bringup", "Preparing the startup catcher.")
    self._progress_card = GreyBigButton("progress", "0%\nstarting")
    self._contract_card = GreyBigButton(
      "RAM-only startup path",
      "No EPS flash writes.\nCancellation is cooperative before PROGRAMMING.\nKeep the vehicle in Park.",
    )
    self._action_button = BigButton("cancel bringup", "swipe down also works")
    self._action_button.set_click_callback(self.dismiss)

    self._scroller.add_widgets([
      self._status_card,
      self._progress_card,
      self._contract_card,
      self._action_button,
    ])


  def show_event(self):
    global _oracle_bringup_active
    super().show_event()
    _oracle_bringup_active = True
    device.set_override_interactive_timeout(300)

  def hide_event(self):
    global _oracle_bringup_active
    _oracle_bringup_active = False
    device.set_override_interactive_timeout(None)

    self._runner.request_cancel()

    super().hide_event()


  def _update_state(self):
    super()._update_state()
    status = self._runner.snapshot()
    stage = status["stage"]
    progress = status["progress"]

    self._status_card.set_text(status["title"])
    self._status_card.set_value(status["detail"])
    self._progress_card.set_value(f"{progress}%\n{STAGE_LABELS.get(stage, stage)}")

    if status["done"]:
      self._action_button.set_text("close")
      self._action_button.set_value("bringup passed")
    elif status["error"]:
      self._action_button.set_text("close")
      self._action_button.set_value("bringup failed")
    elif stage in ("arming", "armed"):
      self._action_button.set_text("cancel bringup")
      self._action_button.set_value("cooperative before PROGRAMMING")
    else:
      self._action_button.set_text("close")
      self._action_button.set_value("bringup continues safely in background")
