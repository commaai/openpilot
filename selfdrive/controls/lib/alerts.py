from cereal import car, log
from common.realtime import DT_CTRL
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.events import EVENTS, EVENT_NAME
import copy

AlertSize = log.ControlsState.AlertSize
AlertStatus = log.ControlsState.AlertStatus
AudibleAlert = car.CarControl.HUDControl.AudibleAlert
VisualAlert = car.CarControl.HUDControl.VisualAlert

class Priority:
  LOWEST = 0
  LOWER = 1
  LOW = 2
  MID = 3
  HIGH = 4
  HIGHEST = 5

class Alert():
  def __init__(self,
               alert_text_1,
               alert_text_2,
               alert_status,
               alert_size,
               alert_priority,
               visual_alert,
               audible_alert,
               duration_sound,
               duration_hud_alert,
               duration_text,
               alert_rate=0.):

    self.alert_type = ""
    self.alert_text_1 = alert_text_1
    self.alert_text_2 = alert_text_2
    self.alert_status = alert_status
    self.alert_size = alert_size
    self.alert_priority = alert_priority
    self.visual_alert = visual_alert
    self.audible_alert = audible_alert

    self.duration_sound = duration_sound
    self.duration_hud_alert = duration_hud_alert
    self.duration_text = duration_text

    self.start_time = 0.
    self.alert_rate = alert_rate

    # typecheck that enums are valid on startup
    tst = car.CarControl.new_message()
    tst.hudControl.visualAlert = self.visual_alert

  def __str__(self):
    return self.alert_text_1 + "/" + self.alert_text_2 + " " + str(self.alert_priority) + "  " + str(
      self.visual_alert) + " " + str(self.audible_alert)

  def __gt__(self, alert2):
    return self.alert_priority > alert2.alert_priority


class AlertManager():

  def __init__(self):
    self.activealerts = []

  def alert_present(self):
    return len(self.activealerts) > 0

  def add(self, frame, alert_type, enabled=True, extra_text_1='', extra_text_2=''):
    self._add(frame, ALERTS[alert_type], alert_type, enabled, extra_text_1, extra_text_2)

  def add_from_event(self, frame, event, event_type, enabled=True, extra_text_1='', extra_text_2=''):
    alert_type = EVENT_NAME[event]
    alert = EVENTS[event][event_type]
    self._add(frame, alert, alert_type, enabled, extra_text_1, extra_text_2)

  def _add(self, frame, alert, alert_type, enabled, extra_text_1, extra_text_2):
    added_alert = copy.copy(alert)
    added_alert.alert_type = alert_type
    added_alert.alert_text_1 += extra_text_1
    added_alert.alert_text_2 += extra_text_2
    added_alert.start_time = frame * DT_CTRL

    # if new alert is higher priority, log it
    if not self.alert_present() or added_alert.alert_priority > self.activealerts[0].alert_priority:
      cloudlog.event('alert_add', alert_type=alert_type, enabled=enabled)

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
