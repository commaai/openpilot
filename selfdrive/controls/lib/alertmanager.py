from cereal import car
from selfdrive.swaglog import cloudlog
import copy


# Priority
class PT:
  HIGH = 3
  MID = 2
  LOW = 1


class Alert(object):
  def __init__(self, 
               alert_text_1,
               alert_text_2,
               alert_priority,
               visual_alert,
               audible_alert, 
               duration_sound,
               duration_hud_alert,
               duration_text):

    self.alert_text_1 = alert_text_1
    self.alert_text_2 = alert_text_2
    self.alert_priority = alert_priority
    self.visual_alert = visual_alert if visual_alert is not None else "none"
    self.audible_alert = audible_alert if audible_alert is not None else "none"
 
    self.duration_sound = duration_sound
    self.duration_hud_alert = duration_hud_alert
    self.duration_text = duration_text

    # typecheck that enums are valid on startup
    tst = car.CarControl.new_message()
    tst.hudControl.visualAlert = self.visual_alert
    tst.hudControl.audibleAlert = self.audible_alert

  def __str__(self):
    return self.alert_text_1 + "/" + self.alert_text_2 + " " + str(self.alert_priority) + "  " + str(
      self.visual_alert) + " " + str(self.audible_alert)

  def __gt__(self, alert2):
    return self.alert_priority > alert2.alert_priority


class AlertManager(object):
  alerts = {

    # Miscellaneous alerts
    "enable": Alert(
        "",
        "",
        PT.MID, None, "beepSingle", .2, 0., 0.),

    "disable": Alert(
        "",
        "",
        PT.MID, None, "beepSingle", .2, 0., 0.),

    "fcw": Alert(
        "", 
        "", 
        PT.LOW, None, None, .1, .1, .1),

    "steerSaturated": Alert(
        "Take Control", 
        "Turn Exceeds Limit", 
        PT.LOW, "steerRequired", "chimeSingle", 1., 2., 3.),

    "steerTempUnavailable": Alert(
        "Take Control", 
        "Steer Temporarily Unavailable", 
        PT.LOW, "steerRequired", "chimeDouble", .4, 2., 3.),

    "preDriverDistracted": Alert(
        "Take Control ", 
        "User Distracted", 
        PT.LOW, "steerRequired", "chimeDouble", .4, 2., 3.),

    "driverDistracted": Alert(
        "Take Control to Regain Speed", 
        "User Distracted", 
        PT.LOW, "steerRequired", "chimeRepeated", .5, .5, .5),

    "startup": Alert(
        "Always Keep Hands on Wheel", 
        "Be Ready to Take Over Any Time", 
        PT.LOW, None, None, 0., 0., 15.),

    "ethicalDilemma": Alert(
        "Take Control Immediately", 
        "Ethical Dilemma Detected", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "steerTempUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Steer Temporary Unavailable", 
        PT.LOW, None, "chimeDouble", .4, 0., 3.),

    # Non-entry only alerts
    "wrongCarModeNoEntry": Alert(
        "Comma Unavailable", 
        "Main Switch Off", 
        PT.LOW, None, "chimeDouble", .4, 0., 3.),

    "dataNeededNoEntry": Alert(
        "Comma Unavailable", 
        "Data needed for calibration. Upload drive, try again", 
        PT.LOW, None, "chimeDouble", .4, 0., 3.),

    "outOfSpaceNoEntry": Alert(
        "Comma Unavailable", 
        "Out of Space", 
        PT.LOW, None, "chimeDouble", .4, 0., 3.),

    "pedalPressedNoEntry": Alert(
        "Comma Unavailable", 
        "Pedal Pressed", 
        PT.LOW, "brakePressed", "chimeDouble", .4, 2., 3.),

    "speedTooLowNoEntry": Alert(
        "Comma Unavailable", 
        "Speed Too Low", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "brakeHoldNoEntry": Alert(
        "Comma Unavailable", 
        "Brake Hold Active", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "parkBrakeNoEntry": Alert(
        "Comma Unavailable", 
        "Park Brake Engaged", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    # Cancellation alerts causing disabling
    "overheat": Alert(
        "Take Control Immediately", 
        "System Overheated", 
        PT.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "wrongGear": Alert(
        "Take Control Immediately", 
        "Gear not D", 
        PT.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "calibrationInvalid": Alert(
        "Take Control Immediately", 
        "Calibration Invalid: Reposition Neo and Recalibrate", 
        PT.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "calibrationInProgress": Alert(
        "Take Control Immediately", 
        "Calibration in Progress",
        PT.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "doorOpen": Alert(
        "Take Control Immediately", 
        "Door Open", 
        PT.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "seatbeltNotLatched": Alert(
        "Take Control Immediately", 
        "Seatbelt Unlatched", 
        PT.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "espDisabled": Alert(
        "Take Control Immediately", 
        "ESP Off", 
        PT.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "radarCommIssue": Alert(
        "Take Control Immediately", 
        "Radar Error: Restart the Car", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "radarFault": Alert(
        "Take Control Immediately", 
        "Radar Error: Restart the Car", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "modelCommIssue": Alert(
        "Take Control Immediately", 
        "Model Error: Restart the Car", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "controlsFailed": Alert(
        "Take Control Immediately", 
        "Controls Failed", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "controlsMismatch": Alert(
        "Take Control Immediately", 
        "Controls Mismatch", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "commIssue": Alert(
        "Take Control Immediately", 
        "CAN Error: Restart the Car", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "steerUnavailable": Alert(
        "Take Control Immediately", 
        "Steer Error: Restart the Car", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "brakeUnavailable": Alert(
        "Take Control Immediately", 
        "Brake Error: Restart the Car", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "gasUnavailable": Alert(
        "Take Control Immediately", 
        "Gas Error: Restart the Car", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "reverseGear": Alert(
        "Take Control Immediately", 
        "Reverse Gear", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "cruiseDisabled": Alert(
        "Take Control Immediately", 
        "Cruise Is Off", 
        PT.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    # not loud cancellations (user is in control)
    "noTarget": Alert(
        "Comma Canceled",
        "No Close Lead", 
        PT.HIGH, None, "chimeDouble", .4, 2., 3.),

    # Cancellation alerts causing non-entry
    "overheatNoEntry": Alert(
        "Comma Unavailable", 
        "System Overheated", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "wrongGearNoEntry": Alert(
        "Comma Unavailable", 
        "Gear not D", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "calibrationInvalidNoEntry": Alert(
        "Comma Unavailable", 
        "Calibration Invalid: Reposition Neo and Recalibrate", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "calibrationInProgressNoEntry": Alert(
        "Comma Unavailable", 
        "Calibration in Progress: ", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "doorOpenNoEntry": Alert(
        "Comma Unavailable", 
        "Door Open", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "seatbeltNotLatchedNoEntry": Alert(
        "Comma Unavailable", 
        "Seatbelt Unlatched", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),
 
    "espDisabledNoEntry": Alert(
        "Comma Unavailable", 
        "ESP Off", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "radarCommIssueNoEntry": Alert(
        "Comma Unavailable", 
        "Radar Error: Restart the Car", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "radarFaultNoEntry": Alert(
        "Comma Unavailable", 
        "Radar Error: Restart the Car", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "modelCommIssueNoEntry": Alert(
        "Comma Unavailable", 
        "Model Error: Restart the Car", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "controlsFailedNoEntry": Alert(
        "Comma Unavailable", 
        "Controls Failed", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "controlsMismatchNoEntry": Alert(
        "Comma Unavailable", 
        "Controls Mismatch", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "commIssueNoEntry": Alert(
        "Comma Unavailable", 
        "CAN Error: Restart the Car", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "steerUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Steer Error: Restart the Car", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "brakeUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Brake Error: Restart the Car", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "gasUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Gas Error: Restart the Car", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "reverseGearNoEntry": Alert(
        "Comma Unavailable", 
        "Reverse Gear", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "cruiseDisabledNoEntry": Alert(
        "Comma Unavailable", 
        "Cruise is Off", 
        PT.LOW, None, "chimeDouble", .4, 2., 3.),

    "noTargetNoEntry": Alert(
        "Comma Unavailable", 
        "No Close Lead",
        PT.LOW, None, "chimeDouble", .4, 2., 3.),
  }

  def __init__(self):
    self.activealerts = []
    self.current_alert = None
    self.add("startup", False)

  def alertPresent(self):
    return len(self.activealerts) > 0

  def add(self, alert_type, enabled=True, extra_text=''):
    alert_type = str(alert_type)
    this_alert = copy.copy(self.alerts[alert_type])
    this_alert.alert_text_2 += extra_text

    # if new alert is higher priority, log it
    if self.current_alert is None or this_alert > self.current_alert:
      cloudlog.event('alert_add',
                     alert_type=alert_type,
                     enabled=enabled)

    self.activealerts.append(this_alert)
    self.activealerts.sort()

  def process_alerts(self, cur_time):
    if self.alertPresent():
      self.alert_start_time = cur_time
      self.current_alert = self.activealerts[0]
      print self.current_alert

    # start with assuming no alerts
    self.alert_text_1 = ""
    self.alert_text_2 = ""
    self.visual_alert = "none"
    self.audible_alert = "none"

    if self.current_alert is not None:
      # ewwwww
      if self.alert_start_time + self.current_alert.duration_sound > cur_time:
        self.audible_alert = self.current_alert.audible_alert

      if self.alert_start_time + self.current_alert.duration_hud_alert > cur_time:
        self.visual_alert = self.current_alert.visual_alert

      if self.alert_start_time + self.current_alert.duration_text > cur_time:
        self.alert_text_1 = self.current_alert.alert_text_1
        self.alert_text_2 = self.current_alert.alert_text_2

      # disable current alert
      if self.alert_start_time + max(self.current_alert.duration_sound, self.current_alert.duration_hud_alert,
                                     self.current_alert.duration_text) < cur_time:
        self.current_alert = None

    # reset
    self.activealerts = []
