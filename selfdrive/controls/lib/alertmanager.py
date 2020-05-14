from cereal import car, log
from common.realtime import DT_CTRL
from selfdrive.swaglog import cloudlog
import copy


AlertSize = log.ControlsState.AlertSize
AlertStatus = log.ControlsState.AlertStatus
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert

class AlertManager():

  def __init__(self):
    self.activealerts = []

  def alert_present(self):
    return len(self.activealerts) > 0

  def add_many(self, frame, alerts, enabled=True):
    for a in alerts:
      self.add(frame, a, enabled=enabled)

  def add(self, frame, alert, enabled=True):
    added_alert = copy.copy(alert)
    added_alert.start_time = frame * DT_CTRL

    # if new alert is higher priority, log it
    if not self.alert_present() or added_alert.alert_priority > self.activealerts[0].alert_priority:
      cloudlog.event('alert_add', alert_type=added_alert.alert_type, enabled=enabled)

    self.activealerts.append(added_alert)

    # sort by priority first and then by start_time
    self.activealerts.sort(key=lambda k: (k.alert_priority, k.start_time), reverse=True)

  def process_alerts(self, frame):
    cur_time = frame * DT_CTRL

    # first get rid of all the expired alerts
    self.activealerts = [a for a in self.activealerts if a.start_time +
                         max(a.duration_sound, a.duration_hud_alert, a.duration_text) > cur_time]

    current_alert = self.activealerts[0] if self.alert_present() else None

    # start with assuming no alerts
    self.alert_type = ""
    self.alert_text_1 = ""
    self.alert_text_2 = ""
    self.alert_status = AlertStatus.normal
    self.alert_size = AlertSize.none
    self.visual_alert = VisualAlert.none
    self.audible_alert = AudibleAlert.none
    self.alert_rate = 0.

    if current_alert:
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
