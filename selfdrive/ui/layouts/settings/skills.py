import time

from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import button_item
from openpilot.system.ui.widgets.scroller_tici import Scroller


def _get_skills() -> dict:
  return Params().get("Skills") or {}


def _run_skill(skill_id: str, skill: dict):
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


class SkillsLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._scroller = None
    self._items: list = []

  def show_event(self):
    super().show_event()
    self._rebuild_items()
    if self._scroller:
      self._scroller.show_event()

  def _rebuild_items(self):
    skills = _get_skills()
    self._items = []

    if not skills:
      no_skills_btn = button_item(
        lambda: "No skills",
        lambda: "",
        description=lambda: "Add via app or API",
        callback=lambda: None,
        enabled=lambda: False,
      )
      self._items.append(no_skills_btn)
    else:
      for skill_id, skill in skills.items():
        btn = button_item(
          lambda s=skill: s["name"],
          lambda: "Run",
          description=lambda s=skill: s["description"],
          callback=lambda sid=skill_id, s=skill: _run_skill(sid, s),
        )
        self._items.append(btn)

    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

  def _render(self, rect):
    if self._scroller:
      self._scroller.render(rect)
