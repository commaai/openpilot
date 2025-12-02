import copy
import os
import json
from collections import defaultdict
from dataclasses import dataclass

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.events import Alert, EmptyAlert


with open(os.path.join(BASEDIR, "selfdrive/selfdrived/alerts_offroad.json")) as f:
  OFFROAD_ALERTS = json.load(f)


def set_offroad_alert(alert: str, show_alert: bool, extra_text: str = None) -> None:
  if show_alert:
    a = copy.copy(OFFROAD_ALERTS[alert])
    a['extra'] = extra_text or ''
    Params().put(alert, a)
  else:
    Params().remove(alert)


@dataclass
class AlertEntry:
  alert: Alert | None = None
  start_frame: int = -1
  end_frame: int = -1
  added_frame: int = -1

  def active(self, frame: int) -> bool:
    return frame <= self.end_frame

  def just_added(self, frame: int) -> bool:
    return self.active(frame) and frame == (self.added_frame + 1)

class AlertManager:
  def __init__(self):
    self.alerts: dict[str, AlertEntry] = defaultdict(AlertEntry)
    self.current_alert = EmptyAlert

  def add_many(self, frame: int, alerts: list[Alert]) -> None:
    for alert in alerts:
      entry = self.alerts[alert.alert_type]
      entry.alert = alert
      if not entry.just_added(frame):
        entry.start_frame = frame
      min_end_frame = entry.start_frame + alert.duration
      entry.end_frame = max(frame + 1, min_end_frame)
      entry.added_frame = frame

  def process_alerts(self, frame: int, clear_event_types: set):
    ae = AlertEntry()
    for v in self.alerts.values():
      if not v.alert:
        continue

      if v.alert.event_type in clear_event_types:
        v.end_frame = -1

      # sort by priority first and then by start_frame
      greater = ae.alert is None or (v.alert.priority, v.start_frame) > (ae.alert.priority, ae.start_frame)
      if v.active(frame) and greater:
        ae = v

    self.current_alert = ae.alert if ae.alert is not None else EmptyAlert
