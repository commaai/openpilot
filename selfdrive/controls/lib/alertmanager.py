import copy
import os
import json
from collections import namedtuple
from typing import List, Dict, Optional

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


AlertEntry = namedtuple('AlertEntry', ['alert', 'start_time'])


class AlertManager:

  def __init__(self):
    self.reset()
    self.activealerts: Dict[str, AlertEntry] = {}

  def reset(self) -> None:
    self.alert_type: str = ""
    self.alert_text_1: str = ""
    self.alert_text_2: str = ""
    self.alert_status = log.ControlsState.AlertStatus.normal
    self.alert_size = log.ControlsState.AlertSize.none
    self.visual_alert = car.CarControl.HUDControl.VisualAlert.none
    self.audible_alert = car.CarControl.HUDControl.AudibleAlert.none
    self.alert_rate: float = 0.

  def add_many(self, frame: int, alerts: List[Alert], enabled: bool = True) -> None:
    for alert in alerts:
      alert_duration = max(alert.duration_sound, alert.duration_hud_alert, alert.duration_text)
      self.activealerts[alert.alert_type] = (alert, frame + int(alert_duration / DT_CTRL))

  def process_alerts(self, frame: int, clear_event_type=None) -> None:
    current_alert = None
    for k, (alert, end_time) in self.activealerts.items():
      if alert.event_type == clear_event_type:
        self.activealerts[k][1] = -1

      # TODO: also sort by time
      # sort by priority first and then by start_time
      active = self.activealerts[k][1] > frame
      if active and (current_alert is None or alert.priority > current_alert.priority):
        current_alert = alert

    print(current_alert)

    # clear current alert
    self.reset()

    if current_alert is not None:
      self.alert_type = current_alert.alert_type
      self.audible_alert = current_alert.audible_alert
      self.visual_alert = current_alert.visual_alert
      self.alert_text_1 = current_alert.alert_text_1
      self.alert_text_2 = current_alert.alert_text_2
      self.alert_status = current_alert.alert_status
      self.alert_size = current_alert.alert_size
      self.alert_rate = current_alert.alert_rate
