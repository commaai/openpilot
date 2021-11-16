import copy
import os
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

from cereal import car, log
from common.basedir import BASEDIR
from common.params import Params
from common.realtime import DT_CTRL
from selfdrive.controls.lib.events import Alert


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


@dataclass
class AlertEntry:
  alert: Optional[Alert] = None
  start_frame: int = -1
  end_frame: int = -1


class AlertManager:

  def __init__(self):
    self.reset()
    self.activealerts: Dict[str, AlertEntry] = defaultdict(AlertEntry)

  def reset(self) -> None:
    self.alert: Optional[Alert] = None
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
      self.activealerts[alert.alert_type].alert = alert
      self.activealerts[alert.alert_type].start_frame = frame
      self.activealerts[alert.alert_type].end_frame = frame + int(alert.duration / DT_CTRL)

  def process_alerts(self, frame: int, clear_event_type=None) -> None:
    current_alert = AlertEntry()
    for k, v in self.activealerts.items():
      if v.alert is None:
        continue

      if v.alert.event_type == clear_event_type:
        self.activealerts[k].end_frame = -1

      # sort by priority first and then by start_frame
      active = self.activealerts[k].end_frame > frame
      greater = current_alert.alert is None or (v.alert.priority, v.start_frame) > (current_alert.alert.priority, current_alert.start_frame)
      if active and greater:
        current_alert = v

    # clear current alert
    self.reset()

    self.alert = current_alert.alert
    if self.alert is not None:
      self.alert_type = self.alert.alert_type
      self.audible_alert = self.alert.audible_alert
      self.visual_alert = self.alert.visual_alert
      self.alert_text_1 = self.alert_text_1
      self.alert_text_2 = self.alert_text_2
      self.alert_status = self.alert_status
      self.alert_size = self.alert_size
      self.alert_rate = self.alert_rate
