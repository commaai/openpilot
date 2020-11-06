import os
import copy
import json
from typing import List, Optional

from cereal import car, log
from common.basedir import BASEDIR
from common.params import Params
from common.realtime import DT_CTRL
from selfdrive.controls.lib.events import Alert
from selfdrive.swaglog import cloudlog


with open(os.path.join(BASEDIR, "selfdrive/controls/lib/alerts_offroad.json")) as f:
  OFFROAD_ALERTS = json.load(f)


def set_offroad_alert(alert: str, show_alert: bool, extra_text: Optional[str] = None) -> None:
  if show_alert:
    a = OFFROAD_ALERTS[alert]
    if extra_text is not None:
      a = copy.copy(OFFROAD_ALERTS[alert])
      a['text'] += extra_text
    Params().put(alert, json.dumps(a))
  else:
    Params().delete(alert)


class AlertManager:

  def __init__(self):
    self.activealerts: List[Alert] = []
    self.clear_current_alert()

  def alert_present(self) -> bool:
    return len(self.activealerts) > 0

  def clear_current_alert(self) -> None:
    self.alert_type: str = ""
    self.alert_text_1: str = ""
    self.alert_text_2: str = ""
    self.alert_status = log.ControlsState.AlertStatus.normal
    self.alert_size = log.ControlsState.AlertSize.none
    self.visual_alert = car.CarControl.HUDControl.VisualAlert.none
    self.audible_alert = car.CarControl.HUDControl.AudibleAlert.none
    self.alert_rate: float = 0.

  def add_many(self, frame: int, alerts: List[Alert], enabled: bool = True) -> None:
    for a in alerts:
      self.add(frame, a, enabled=enabled)

  def add(self, frame: int, alert: Alert, enabled: bool = True) -> None:
    added_alert = copy.copy(alert)
    added_alert.start_time = frame * DT_CTRL

    # if new alert is higher priority, log it
    if not self.alert_present() or added_alert.alert_priority > self.activealerts[0].alert_priority:
      cloudlog.event('alert_add', alert_type=added_alert.alert_type, enabled=enabled)

    self.activealerts.append(added_alert)

    # sort by priority first and then by start_time
    self.activealerts.sort(key=lambda k: (k.alert_priority, k.start_time), reverse=True)

  def process_alerts(self, frame: int) -> None:
    cur_time = frame * DT_CTRL

    # first get rid of all the expired alerts
    self.activealerts = [a for a in self.activealerts if a.start_time +
                         max(a.duration_sound, a.duration_hud_alert, a.duration_text) > cur_time]

    # start with assuming no alerts
    self.clear_current_alert()

    current_alert = self.activealerts[0] if self.alert_present() else None
    if current_alert is not None:
      self.alert_type = current_alert.alert_type

      if current_alert.start_time + current_alert.duration_sound > cur_time:
        self.audible_alert = current_alert.audible_alert

      if current_alert.start_time + current_alert.duration_hud_alert > cur_time:
        self.visual_alert = current_alert.visual_alert

      if current_alert.start_time + current_alert.duration_text > cur_time:
        self.alert_text_1 = current_alert.alert_text_1
        self.alert_text_2 = current_alert.alert_text_2
        self.alert_status = current_alert.alert_status
        self.alert_size = current_alert.alert_size
        self.alert_rate = current_alert.alert_rate
