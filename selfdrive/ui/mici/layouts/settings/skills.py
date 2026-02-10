from collections.abc import Callable

from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets import NavWidget
from openpilot.selfdrive.ui.mici.widgets.button import BigButton


def _get_skills() -> dict:
  return Params().get("Skills") or {}


def _run_skill(skill_id: str, skill: dict):
  import time
  from panda import Panda
  from opendbc.car.structs import CarParams

  msg_id = skill["msg_id"]
  bus = skill["bus"]
  count = skill["count"]
  data = bytes.fromhex(skill["data"])

  panda = Panda()
  panda.set_safety_mode(CarParams.SafetyModel.allOutput)

  for _ in range(count):
    panda.can_send(msg_id, data, bus)
    time.sleep(0.02)

  panda.set_safety_mode(CarParams.SafetyModel.silent)
  panda.close()


class SkillsLayoutMici(NavWidget):
  def __init__(self, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)
    self._params = Params()
    self._scroller = None
    self._buttons: list[BigButton] = []

  def show_event(self):
    super().show_event()
    self._rebuild_buttons()
    if self._scroller:
      self._scroller.show_event()

  def _rebuild_buttons(self):
    skills = _get_skills()
    self._buttons = []

    if not skills:
      no_skills_btn = BigButton("no skills", "add via app or api")
      no_skills_btn.set_enabled(False)
      self._buttons.append(no_skills_btn)
    else:
      for skill_id, skill in skills.items():
        btn = BigButton(skill["name"], skill["description"])
        btn.set_click_callback(lambda sid=skill_id, s=skill: _run_skill(sid, s))
        self._buttons.append(btn)

    self._scroller = Scroller(self._buttons, snap_items=False)

  def _render(self, rect):
    if self._scroller:
      self._scroller.render(rect)
