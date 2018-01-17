from cereal import car, log
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
import copy


# Priority
class Priority:
  HIGH = 3
  MID = 2
  LOW = 1
  LOWEST = 0

AlertSize = log.Live100Data.AlertSize
AlertStatus = log.Live100Data.AlertStatus

class Alert(object):
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
               duration_text):

    self.alert_text_1 = alert_text_1
    self.alert_text_2 = alert_text_2
    self.alert_status = alert_status
    self.alert_size = alert_size
    self.alert_priority = alert_priority
    self.visual_alert = visual_alert if visual_alert is not None else "none"
    self.audible_alert = audible_alert if audible_alert is not None else "none"
 
    self.duration_sound = duration_sound
    self.duration_hud_alert = duration_hud_alert
    self.duration_text = duration_text

    self.start_time = 0.

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
        AlertStatus.normal, AlertSize.none,
        Priority.MID, None, "beepSingle", .2, 0., 0.),

    "disable": Alert(
        "",
        "",
        AlertStatus.normal, AlertSize.none,
        Priority.MID, None, "beepSingle", .2, 0., 0.),

    "fcw": Alert(
        "Brake!", 
        "Risk of Collision", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "fcw", "chimeRepeated", 1., 2., 2.),

    "steerSaturated": Alert(
        "Take Control", 
        "Turn Exceeds Limit", 
        AlertStatus.userPrompt, AlertSize.full,
        Priority.LOW, "steerRequired", "chimeSingle", 1., 2., 3.),

    "steerTempUnavailable": Alert(
        "Take Control", 
        "Steer Temporarily Unavailable", 
        AlertStatus.userPrompt, AlertSize.full,
        Priority.LOW, "steerRequired", "chimeDouble", .4, 2., 3.),

    "preDriverDistracted": Alert(
        "Take Control", 
        "User Distracted", 
        AlertStatus.userPrompt, AlertSize.full,
        Priority.LOW, "steerRequired", "chimeDouble", .1, .1, .1),

    "driverDistracted": Alert(
        "Take Control to Regain Speed", 
        "User Distracted", 
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", .1, .1, .1),

    "startup": Alert(
        "Always Keep Hands on Wheel", 
        "Be Ready to Take Over Any Time", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOWEST, None, None, 0., 0., 15.),

    "ethicalDilemma": Alert(
        "Take Control Immediately", 
        "Ethical Dilemma Detected", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "steerTempUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Steer Temporary Unavailable", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 0., 3.),

    "manualRestart": Alert(
        "Take Control",
        "Resume Driving Manually",
        AlertStatus.userPrompt, AlertSize.full,
        Priority.LOW, None, None, 0., 0., .2),

    # Non-entry only alerts
    "wrongCarModeNoEntry": Alert(
        "Comma Unavailable", 
        "Main Switch Off", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 0., 3.),

    "dataNeededNoEntry": Alert(
        "Comma Unavailable", 
        "Data needed for calibration. Upload drive, try again", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 0., 3.),

    "outOfSpaceNoEntry": Alert(
        "Comma Unavailable", 
        "Out of Space", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 0., 3.),

    "pedalPressedNoEntry": Alert(
        "Comma Unavailable", 
        "Pedal Pressed", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, "brakePressed", "chimeDouble", .4, 2., 3.),

    "speedTooLowNoEntry": Alert(
        "Comma Unavailable", 
        "Speed Too Low", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "brakeHoldNoEntry": Alert(
        "Comma Unavailable", 
        "Brake Hold Active", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "parkBrakeNoEntry": Alert(
        "Comma Unavailable", 
        "Park Brake Engaged", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "lowSpeedLockoutNoEntry": Alert(
        "Comma Unavailable",
        "Cruise Fault: Restart the Car",
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    # Cancellation alerts causing soft disabling
    "overheat": Alert(
        "Take Control Immediately", 
        "System Overheated", 
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "wrongGear": Alert(
        "Take Control Immediately", 
        "Gear not D", 
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "calibrationInvalid": Alert(
        "Take Control Immediately", 
        "Calibration Invalid: Reposition EON and Recalibrate", 
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "calibrationInProgress": Alert(
        "Take Control Immediately", 
        "Calibration in Progress",
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "doorOpen": Alert(
        "Take Control Immediately", 
        "Door Open", 
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "seatbeltNotLatched": Alert(
        "Take Control Immediately", 
        "Seatbelt Unlatched", 
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    "espDisabled": Alert(
        "Take Control Immediately", 
        "ESP Off", 
        AlertStatus.critical, AlertSize.full,
        Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

    # Cancellation alerts causing immediate disabling
    "radarCommIssue": Alert(
        "Take Control Immediately", 
        "Radar Error: Restart the Car", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "radarFault": Alert(
        "Take Control Immediately", 
        "Radar Error: Restart the Car", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "modelCommIssue": Alert(
        "Take Control Immediately", 
        "Model Error: Restart the Car", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "controlsFailed": Alert(
        "Take Control Immediately", 
        "Controls Failed", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "controlsMismatch": Alert(
        "Take Control Immediately", 
        "Controls Mismatch", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "commIssue": Alert(
        "Take Control Immediately", 
        "CAN Error: Restart the Car", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "steerUnavailable": Alert(
        "Take Control Immediately", 
        "Steer Fault: Restart the Car", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "brakeUnavailable": Alert(
        "Take Control Immediately", 
        "Brake Fault: Restart the Car", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "gasUnavailable": Alert(
        "Take Control Immediately", 
        "Gas Fault: Restart the Car", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "reverseGear": Alert(
        "Take Control Immediately", 
        "Reverse Gear", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    "cruiseDisabled": Alert(
        "Take Control Immediately", 
        "Cruise Is Off", 
        AlertStatus.critical, AlertSize.full,
        Priority.HIGH, "steerRequired", "chimeRepeated", 1., 3., 4.),

    # not loud cancellations (user is in control)
    "noTarget": Alert(
        "Comma Canceled",
        "No Close Lead", 
        AlertStatus.normal, AlertSize.full,
        Priority.HIGH, None, "chimeDouble", .4, 2., 3.),

    "speedTooLow": Alert(
        "Comma Canceled",
        "Speed Too Low", 
        AlertStatus.normal, AlertSize.full,
        Priority.HIGH, None, "chimeDouble", .4, 2., 3.),

    # Cancellation alerts causing non-entry
    "overheatNoEntry": Alert(
        "Comma Unavailable", 
        "System Overheated", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "wrongGearNoEntry": Alert(
        "Comma Unavailable", 
        "Gear not D", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "calibrationInvalidNoEntry": Alert(
        "Comma Unavailable", 
        "Calibration Invalid: Reposition EON and Recalibrate", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "calibrationInProgressNoEntry": Alert(
        "Comma Unavailable", 
        "Calibration in Progress", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "doorOpenNoEntry": Alert(
        "Comma Unavailable", 
        "Door Open", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "seatbeltNotLatchedNoEntry": Alert(
        "Comma Unavailable", 
        "Seatbelt Unlatched", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),
 
    "espDisabledNoEntry": Alert(
        "Comma Unavailable", 
        "ESP Off", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "radarCommIssueNoEntry": Alert(
        "Comma Unavailable", 
        "Radar Error: Restart the Car", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "radarFaultNoEntry": Alert(
        "Comma Unavailable", 
        "Radar Error: Restart the Car", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "modelCommIssueNoEntry": Alert(
        "Comma Unavailable", 
        "Model Error: Restart the Car", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "controlsFailedNoEntry": Alert(
        "Comma Unavailable", 
        "Controls Failed", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "commIssueNoEntry": Alert(
        "Comma Unavailable", 
        "CAN Error: Restart the Car", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "steerUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Steer Fault: Restart the Car", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "brakeUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Brake Fault: Restart the Car", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "gasUnavailableNoEntry": Alert(
        "Comma Unavailable", 
        "Gas Error: Restart the Car", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "reverseGearNoEntry": Alert(
        "Comma Unavailable", 
        "Reverse Gear", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "cruiseDisabledNoEntry": Alert(
        "Comma Unavailable", 
        "Cruise is Off", 
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    "noTargetNoEntry": Alert(
        "Comma Unavailable", 
        "No Close Lead",
        AlertStatus.normal, AlertSize.full,
        Priority.LOW, None, "chimeDouble", .4, 2., 3.),

    # permanent alerts to display on small UI upper box
    "steerUnavailablePermanent": Alert(
        "STEER FAULT", 
        "RESTART THE CAR",
        AlertStatus.normal, AlertSize.small,
        Priority.LOWEST, None, None, 0., 0., .2),

    "brakeUnavailablePermanent": Alert(
        "BRAKE FAULT", 
        "RESTART THE CAR",
        AlertStatus.normal, AlertSize.small,
        Priority.LOWEST, None, None, 0., 0., .2),

    "lowSpeedLockoutPermanent": Alert(
        "CRUISE FAULT", 
        "RESTART THE CAR",
        AlertStatus.normal, AlertSize.small,
        Priority.LOWEST, None, None, 0., 0., .2),
  }

  def __init__(self):
    self.activealerts = []

  def alertPresent(self):
    return len(self.activealerts) > 0

  def add(self, alert_type, enabled=True, extra_text=''):
    alert_type = str(alert_type)
    added_alert = copy.copy(self.alerts[alert_type])
    added_alert.alert_text_2 += extra_text
    added_alert.start_time = sec_since_boot()

    # if new alert is higher priority, log it
    if not self.alertPresent() or \
       added_alert.alert_priority > self.activealerts[0].alert_priority:
      cloudlog.event('alert_add',
                     alert_type=alert_type,
                     enabled=enabled)

    self.activealerts.append(added_alert)
    self.activealerts.sort(key=lambda k: k.alert_priority, reverse=True)

  # TODO: cycle through alerts?
  def process_alerts(self, cur_time):

    # first get rid of all the expired alerts
    self.activealerts = [a for a in self.activealerts if a.start_time + 
                         max(a.duration_sound, a.duration_hud_alert, a.duration_text) > cur_time]

    ca = self.activealerts[0] if self.alertPresent() else None

    # start with assuming no alerts
    self.alert_text_1 = ""
    self.alert_text_2 = ""
    self.alert_status = AlertStatus.normal
    self.alert_size = AlertSize.none
    self.visual_alert = "none"
    self.audible_alert = "none"

    if ca:
      if ca.start_time + ca.duration_sound > cur_time:
        self.audible_alert = ca.audible_alert

      if ca.start_time + ca.duration_hud_alert > cur_time:
        self.visual_alert = ca.visual_alert

      if ca.start_time + ca.duration_text > cur_time:
        self.alert_text_1 = ca.alert_text_1
        self.alert_text_2 = ca.alert_text_2
        self.alert_status = ca.alert_status
        self.alert_size = ca.alert_size
