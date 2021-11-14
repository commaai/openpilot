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
from selfdrive.controls.lib.events import EVENTS, ET


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

  def SA_set_frame(self, frame):
    self.SA_frame = frame

  def SA_set_enabled(self, enabled):
    self.SA_enabled = enabled

  def SA_add(self, alert_name, extra_text_1='', extra_text_2=''):
    alert = EVENTS[alert_name][ET.PERMANENT]  # assume permanent (to display in all states)
    added_alert = copy.copy(alert)
    added_alert.start_time = self.SA_frame * DT_CTRL
    added_alert.alert_text_1 += extra_text_1
    added_alert.alert_text_2 += extra_text_2
    added_alert.alert_type = f"{alert_name}/{ET.PERMANENT}"  # fixes alerts being silent
    added_alert.event_type = ET.PERMANENT

    self.activealerts[alert.alert_type].alert = added_alert
    self.activealerts[alert.alert_type].start_frame = self.SA_frame
    self.activealerts[alert.alert_type].end_frame = self.SA_frame + int(alert.duration / DT_CTRL)

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

    a = current_alert.alert
    if a is not None:
      self.alert_type = a.alert_type
      self.audible_alert = a.audible_alert
      self.visual_alert = a.visual_alert
      self.alert_text_1 = a.alert_text_1
      self.alert_text_2 = a.alert_text_2
      self.alert_status = a.alert_status
      self.alert_size = a.alert_size
      self.alert_rate = a.alert_rate
