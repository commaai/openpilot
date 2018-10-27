from cereal import car, log
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
import copy


# Priority
class Priority:
  LOWEST = 0
  LOW_LOWEST = 1
  LOW = 2
  MID = 3
  HIGH = 4
  HIGHEST = 5

AlertSize = log.Live100Data.AlertSize
AlertStatus = log.Live100Data.AlertStatus

class Alert(object):
  def __init__(self,
               alert_type,
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

    self.alert_type = alert_type
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
    self.alert_rate = alert_rate

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
    alert.alert_type: alert
    for alert in [
        # Miscellaneous alerts
        Alert(
            "enable",
            "",
            "",
            AlertStatus.normal, AlertSize.none,
            Priority.MID, None, "beepSingle", .2, 0., 0.),

        Alert(
            "disable",
            "",
            "",
            AlertStatus.normal, AlertSize.none,
            Priority.MID, None, "beepSingle", .2, 0., 0.),

        Alert(
            "fcw",
            "BRAKE!",
            "Risk of Collision",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "fcw", "chimeRepeated", 1., 2., 2.),

        Alert(
            "steerSaturated",
            "TAKE CONTROL",
            "Turn Exceeds Steering Limit",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.LOW, "steerRequired", "chimeSingle", 1., 2., 3.),

        Alert(
            "steerTempUnavailable",
            "TAKE CONTROL",
            "Steering Temporarily Unavailable",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.LOW, "steerRequired", "chimeDouble", .4, 2., 3.),

        Alert(
            "steerTempUnavailableMute",
            "TAKE CONTROL",
            "Steering Temporarily Unavailable",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.LOW, None, None, .2, .2, .2),

        Alert(
            "preDriverDistracted",
            "KEEP EYES ON ROAD: User Appears Distracted",
            "",
            AlertStatus.normal, AlertSize.small,
            Priority.LOW, "steerRequired", None, 0., .1, .1, alert_rate=0.75),

        Alert(
            "promptDriverDistracted",
            "KEEP EYES ON ROAD",
            "User Appears Distracted",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.MID, "steerRequired", "chimeRepeated", .1, .1, .1),

        Alert(
            "driverDistracted",
            "DISENGAGE IMMEDIATELY",
            "User Was Distracted",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGH, "steerRequired", "chimeRepeated", .1, .1, .1),

        Alert(
            "preDriverUnresponsive",
            "TOUCH STEERING WHEEL: No Driver Monitoring",
            "",
            AlertStatus.normal, AlertSize.small,
            Priority.LOW, "steerRequired", None, 0., .1, .1, alert_rate=0.75),

        Alert(
            "promptDriverUnresponsive",
            "TOUCH STEERING WHEEL",
            "User Is Unresponsive",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.MID, "steerRequired", "chimeRepeated", .1, .1, .1),

        Alert(
            "driverUnresponsive",
            "DISENGAGE IMMEDIATELY",
            "User Was Unresponsive",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGH, "steerRequired", "chimeRepeated", .1, .1, .1),

        Alert(
            "driverMonitorOff",
            "DRIVER MONITOR IS UNAVAILABLE",
            "Accuracy Is Low",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, None, .4, 0., 4.),

        Alert(
            "driverMonitorOn",
            "DRIVER MONITOR IS AVAILABLE",
            "Accuracy Is High",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, None, .4, 0., 4.),

        Alert(
            "geofence",
            "DISENGAGEMENT REQUIRED",
            "Not in Geofenced Area",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.HIGH, "steerRequired", "chimeRepeated", .1, .1, .1),

        Alert(
            "startup",
            "Be ready to take over at any time",
            "Always keep hands on wheel and eyes on road",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW_LOWEST, None, None, 0., 0., 15.),

        Alert(
            "ethicalDilemma",
            "TAKE CONTROL IMMEDIATELY",
            "Ethical Dilemma Detected",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "steerTempUnavailableNoEntry",
            "openpilot Unavailable",
            "Steering Temporarily Unavailable",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 0., 3.),

        Alert(
            "manualRestart",
            "TAKE CONTROL",
            "Resume Driving Manually",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.LOW, None, None, 0., 0., .2),

        Alert(
            "resumeRequired",
            "STOPPED",
            "Press Resume to Move",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.LOW, None, None, 0., 0., .2),

        Alert(
            "belowSteerSpeed",
            "TAKE CONTROL",
            "Steer Unavailable Below ",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.MID, "steerRequired", None, 0., 0., .1),

        Alert(
            "debugAlert",
            "DEBUG ALERT",
            "",
            AlertStatus.userPrompt, AlertSize.mid,
            Priority.LOW, None, None, .1, .1, .1),

        # Non-entry only alerts
        Alert(
            "wrongCarModeNoEntry",
            "openpilot Unavailable",
            "Main Switch Off",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 0., 3.),

        Alert(
            "dataNeededNoEntry",
            "openpilot Unavailable",
            "Data Needed for Calibration. Upload Drive, Try Again",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 0., 3.),

        Alert(
            "outOfSpaceNoEntry",
            "openpilot Unavailable",
            "Out of Storage Space",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 0., 3.),

        Alert(
            "pedalPressedNoEntry",
            "openpilot Unavailable",
            "Pedal Pressed During Attempt",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, "brakePressed", "chimeDouble", .4, 2., 3.),

        Alert(
            "speedTooLowNoEntry",
            "openpilot Unavailable",
            "Speed Too Low",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "brakeHoldNoEntry",
            "openpilot Unavailable",
            "Brake Hold Active",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "parkBrakeNoEntry",
            "openpilot Unavailable",
            "Park Brake Engaged",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "lowSpeedLockoutNoEntry",
            "openpilot Unavailable",
            "Cruise Fault: Restart the Car",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "lowBatteryNoEntry",
            "openpilot Unavailable",
            "Low Battery",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        # Cancellation alerts causing soft disabling
        Alert(
            "overheat",
            "TAKE CONTROL IMMEDIATELY",
            "System Overheated",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "wrongGear",
            "TAKE CONTROL IMMEDIATELY",
            "Gear not D",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "calibrationInvalid",
            "TAKE CONTROL IMMEDIATELY",
            "Calibration Invalid: Reposition EON and Recalibrate",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "calibrationIncomplete",
            "TAKE CONTROL IMMEDIATELY",
            "Calibration in Progress",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "doorOpen",
            "TAKE CONTROL IMMEDIATELY",
            "Door Open",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "seatbeltNotLatched",
            "TAKE CONTROL IMMEDIATELY",
            "Seatbelt Unlatched",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "espDisabled",
            "TAKE CONTROL IMMEDIATELY",
            "ESP Off",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        Alert(
            "lowBattery",
            "TAKE CONTROL IMMEDIATELY",
            "Low Battery",
            AlertStatus.critical, AlertSize.full,
            Priority.MID, "steerRequired", "chimeRepeated", 1., 3., 3.),

        # Cancellation alerts causing immediate disabling
        Alert(
            "radarCommIssue",
            "TAKE CONTROL IMMEDIATELY",
            "Radar Error: Restart the Car",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "radarFault",
            "TAKE CONTROL IMMEDIATELY",
            "Radar Error: Restart the Car",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "modelCommIssue",
            "TAKE CONTROL IMMEDIATELY",
            "Model Error: Check Internet Connection",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "controlsFailed",
            "TAKE CONTROL IMMEDIATELY",
            "Controls Failed",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "controlsMismatch",
            "TAKE CONTROL IMMEDIATELY",
            "Controls Mismatch",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "commIssue",
            "TAKE CONTROL IMMEDIATELY",
            "CAN Error: Check Connections",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "steerUnavailable",
            "TAKE CONTROL IMMEDIATELY",
            "LKAS Fault: Restart the Car",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "brakeUnavailable",
            "TAKE CONTROL IMMEDIATELY",
            "Cruise Fault: Restart the Car",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "gasUnavailable",
            "TAKE CONTROL IMMEDIATELY",
            "Gas Fault: Restart the Car",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "reverseGear",
            "TAKE CONTROL IMMEDIATELY",
            "Reverse Gear",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "cruiseDisabled",
            "TAKE CONTROL IMMEDIATELY",
            "Cruise Is Off",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        Alert(
            "plannerError",
            "TAKE CONTROL IMMEDIATELY",
            "Planner Solution Error",
            AlertStatus.critical, AlertSize.full,
            Priority.HIGHEST, "steerRequired", "chimeRepeated", 1., 3., 4.),

        # not loud cancellations (user is in control)
        Alert(
            "noTarget",
            "openpilot Canceled",
            "No close lead car",
            AlertStatus.normal, AlertSize.mid,
            Priority.HIGH, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "speedTooLow",
            "openpilot Canceled",
            "Speed too low",
            AlertStatus.normal, AlertSize.mid,
            Priority.HIGH, None, "chimeDouble", .4, 2., 3.),

        # Cancellation alerts causing non-entry
        Alert(
            "overheatNoEntry",
            "openpilot Unavailable",
            "System overheated",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "wrongGearNoEntry",
            "openpilot Unavailable",
            "Gear not D",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "calibrationInvalidNoEntry",
            "openpilot Unavailable",
            "Calibration Invalid: Reposition EON and Recalibrate",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "calibrationIncompleteNoEntry",
            "openpilot Unavailable",
            "Calibration in Progress",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "doorOpenNoEntry",
            "openpilot Unavailable",
            "Door open",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "seatbeltNotLatchedNoEntry",
            "openpilot Unavailable",
            "Seatbelt unlatched",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "espDisabledNoEntry",
            "openpilot Unavailable",
            "ESP Off",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "geofenceNoEntry",
            "openpilot Unavailable",
            "Not in Geofenced Area",
            AlertStatus.normal, AlertSize.mid,
            Priority.MID, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "radarCommIssueNoEntry",
            "openpilot Unavailable",
            "Radar Error: Restart the Car",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "radarFaultNoEntry",
            "openpilot Unavailable",
            "Radar Error: Restart the Car",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "modelCommIssueNoEntry",
            "openpilot Unavailable",
            "Model Error: Check Internet Connection",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "controlsFailedNoEntry",
            "openpilot Unavailable",
            "Controls Failed",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "commIssueNoEntry",
            "openpilot Unavailable",
            "CAN Error: Check Connections",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "steerUnavailableNoEntry",
            "openpilot Unavailable",
            "LKAS Fault: Restart the Car",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "brakeUnavailableNoEntry",
            "openpilot Unavailable",
            "Cruise Fault: Restart the Car",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "gasUnavailableNoEntry",
            "openpilot Unavailable",
            "Gas Error: Restart the Car",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "reverseGearNoEntry",
            "openpilot Unavailable",
            "Reverse Gear",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "cruiseDisabledNoEntry",
            "openpilot Unavailable",
            "Cruise is Off",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "noTargetNoEntry",
            "openpilot Unavailable",
            "No Close Lead Car",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        Alert(
            "plannerErrorNoEntry",
            "openpilot Unavailable",
            "Planner Solution Error",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOW, None, "chimeDouble", .4, 2., 3.),

        # permanent alerts
        Alert(
            "steerUnavailablePermanent",
            "LKAS Fault: Restart the car to engage",
            "",
            AlertStatus.normal, AlertSize.small,
            Priority.LOW_LOWEST, None, None, 0., 0., .2),

        Alert(
            "brakeUnavailablePermanent",
            "Cruise Fault: Restart the car to engage",
            "",
            AlertStatus.normal, AlertSize.small,
            Priority.LOW_LOWEST, None, None, 0., 0., .2),

        Alert(
            "lowSpeedLockoutPermanent",
            "Cruise Fault: Restart the car to engage",
            "",
            AlertStatus.normal, AlertSize.small,
            Priority.LOW_LOWEST, None, None, 0., 0., .2),

        Alert(
            "calibrationIncompletePermanent",
            "Calibration in Progress: ",
            "Drive Above ",
            AlertStatus.normal, AlertSize.mid,
            Priority.LOWEST, None, None, 0., 0., .2),
    ]
  }

  def __init__(self):
    self.activealerts = []

  def alertPresent(self):
    return len(self.activealerts) > 0

  def add(self, alert_type, enabled=True, extra_text_1='', extra_text_2=''):
    alert_type = str(alert_type)
    added_alert = copy.copy(self.alerts[alert_type])
    added_alert.alert_text_1 += extra_text_1
    added_alert.alert_text_2 += extra_text_2
    added_alert.start_time = sec_since_boot()

    # if new alert is higher priority, log it
    if not self.alertPresent() or \
       added_alert.alert_priority > self.activealerts[0].alert_priority:
      cloudlog.event('alert_add',
                     alert_type=alert_type,
                     enabled=enabled)

    self.activealerts.append(added_alert)
    # sort by priority first and then by start_time
    self.activealerts.sort(key=lambda k: (k.alert_priority, k.start_time), reverse=True)

  # TODO: cycle through alerts?
  def process_alerts(self, cur_time):

    # first get rid of all the expired alerts
    self.activealerts = [a for a in self.activealerts if a.start_time +
                         max(a.duration_sound, a.duration_hud_alert, a.duration_text) > cur_time]

    ca = self.activealerts[0] if self.alertPresent() else None

    # start with assuming no alerts
    self.alert_type = ""
    self.alert_text_1 = ""
    self.alert_text_2 = ""
    self.alert_status = AlertStatus.normal
    self.alert_size = AlertSize.none
    self.visual_alert = "none"
    self.audible_alert = "none"
    self.alert_rate = 0.

    if ca:
      if ca.start_time + ca.duration_sound > cur_time:
        self.audible_alert = ca.audible_alert

      if ca.start_time + ca.duration_hud_alert > cur_time:
        self.visual_alert = ca.visual_alert

      if ca.start_time + ca.duration_text > cur_time:
        self.alert_type = ca.alert_type
        self.alert_text_1 = ca.alert_text_1
        self.alert_text_2 = ca.alert_text_2
        self.alert_status = ca.alert_status
        self.alert_size = ca.alert_size
        self.alert_rate = ca.alert_rate
