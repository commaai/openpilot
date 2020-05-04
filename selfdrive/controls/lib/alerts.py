from cereal import car, log

# Priority
class Priority:
  LOWEST = 0
  LOWER = 1
  LOW = 2
  MID = 3
  HIGH = 4
  HIGHEST = 5

AlertSize = log.ControlsState.AlertSize
AlertStatus = log.ControlsState.AlertStatus
AudibleAlert = car.CarControl.HUDControl.AudibleAlert
VisualAlert = car.CarControl.HUDControl.VisualAlert

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

class NoEntryAlert(Alert):
  def __init__(self, alert_text_2, audible_alert=AudibleAlert.chimeError,
               visual_alert=VisualAlert.none, duration_hud_alert=2.):
    super().__init__("openpilot Unavailable", alert_text_2, AlertStatus.normal, \
          AlertSize.mid, Priority.LOW, visual_alert, \
          audible_alert, .4, duration_hud_alert, 3.)


# TODO: make PermanentAlert, etc.

ALERTS = {
  # Miscellaneous alerts
  "enable": Alert(
      "",
      "",
      AlertStatus.normal, AlertSize.none,
      Priority.MID, VisualAlert.none, AudibleAlert.chimeEngage, .2, 0., 0.),

  "disable": Alert(
      "",
      "",
      AlertStatus.normal, AlertSize.none,
      Priority.MID, VisualAlert.none, AudibleAlert.chimeDisengage, .2, 0., 0.),

  "fcw": Alert(
      "BRAKE!",
      "Risk of Collision",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.chimeWarningRepeat, 1., 2., 2.),

  "fcwStock": Alert(
      "BRAKE!",
      "Risk of Collision",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.none, 1., 2., 2.),  # no EON chime for stock FCW

  "steerSaturated": Alert(
      "TAKE CONTROL",
      "Turn Exceeds Steering Limit",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, 1., 2., 3.),

  "steerTempUnavailable": Alert(
      "TAKE CONTROL",
      "Steering Temporarily Unavailable",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),

  "steerTempUnavailableMute": Alert(
      "TAKE CONTROL",
      "Steering Temporarily Unavailable",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .2, .2, .2),

  "preDriverDistracted": Alert(
      "KEEP EYES ON ROAD: Driver Distracted",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),

  "promptDriverDistracted": Alert(
      "KEEP EYES ON ROAD",
      "Driver Appears Distracted",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),

  "driverDistracted": Alert(
      "DISENGAGE IMMEDIATELY",
      "Driver Was Distracted",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),

  "preDriverUnresponsive": Alert(
      "TOUCH STEERING WHEEL: No Face Detected",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),

  "promptDriverUnresponsive": Alert(
      "TOUCH STEERING WHEEL",
      "Driver Is Unresponsive",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),

  "driverUnresponsive": Alert(
      "DISENGAGE IMMEDIATELY",
      "Driver Was Unresponsive",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),

  "driverMonitorLowAcc": Alert(
      "CHECK DRIVER FACE VISIBILITY",
      "Driver Monitor Model Output Uncertain",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .4, 0., 1.),

  "geofence": Alert(
      "",
      "DISENGAGEMENT REQUIRED",
      "Not in Geofenced Area",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),

  "startup": Alert(
      "Be ready to take over at any time",
      "Always keep hands on wheel and eyes on road",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),

  "startupMaster": Alert(
      "WARNING: This branch is not tested",
      "Always keep hands on wheel and eyes on road",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),

  "startupNoControl": Alert(
      "Dashcam mode",
      "Always keep hands on wheel and eyes on road",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),

  "startupNoCar": Alert(
      "Dashcam mode for unsupported car",
      "Always keep hands on wheel and eyes on road",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),

  "ethicalDilemma": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Ethical Dilemma Detected",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 1., 3., 3.),

  "steerTempUnavailableNoEntry": NoEntryAlert("Steering Temporarily Unavailable",
                                              duration_hud_alert=0.),

  "manualRestart": Alert(
      "TAKE CONTROL",
      "Resume Driving Manually",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "resumeRequired": Alert(
      "STOPPED",
      "Press Resume to Move",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "belowSteerSpeed": Alert(
      "TAKE CONTROL",
      "Steer Unavailable Below ",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.none, 0., 0.4, .3),

  "debugAlert": Alert(
      "DEBUG ALERT",
      "",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .1, .1, .1),

  "preLaneChangeLeft": Alert(
      "Steer Left to Start Lane Change",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),

  "preLaneChangeRight": Alert(
      "Steer Right to Start Lane Change",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),

  "laneChange": Alert(
      "Changing Lane",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1),

  "posenetInvalid": Alert(
      "TAKE CONTROL",
      "Vision Model Output Uncertain",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),

  # Non-entry only alerts
  "wrongCarModeNoEntry": NoEntryAlert("Main Switch Off",
                                      duration_hud_alert=0.),
  "dataNeededNoEntry": NoEntryAlert("Calibration Needs Data. Upload Drive, Try Again",
                                    duration_hud_alert=0.),
  "outOfSpaceNoEntry": NoEntryAlert("Out of Storage Space",
                                    duration_hud_alert=0.),
  "pedalPressedNoEntry": NoEntryAlert("Pedal Pressed During Attempt",
                                      visual_alert=VisualAlert.brakePressed),
  "speedTooLowNoEntry": NoEntryAlert("Speed Too Low"),
  "brakeHoldNoEntry": NoEntryAlert("Brake Hold Active"),
  "parkBrakeNoEntry": NoEntryAlert("Park Brake Engaged"),
  "lowSpeedLockoutNoEntry": NoEntryAlert("Cruise Fault: Restart the Car"),
  "lowBatteryNoEntry": NoEntryAlert("Low Battery"),
  "sensorDataInvalidNoEntry": NoEntryAlert("No Data from Device Sensors"),
  "soundsUnavailableNoEntry": NoEntryAlert("Speaker not found"),
  "tooDistractedNoEntry": NoEntryAlert("Distraction Level Too High"),

  # Cancellation alerts causing soft disabling
  "overheat": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "System Overheated",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "wrongGear": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Gear not D",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "calibrationInvalid": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Calibration Invalid: Reposition Device and Recalibrate",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "calibrationIncomplete": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Calibration in Progress",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "doorOpen": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Door Open",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "seatbeltNotLatched": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Seatbelt Unlatched",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "espDisabled": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "ESP Off",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "lowBattery": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Low Battery",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "commIssue": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Communication Issue between Processes",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "radarCommIssue": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Radar Communication Issue",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "radarCanError": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Radar Error: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  "radarFault": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Radar Error: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),


  "lowMemory": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Low Memory: Reboot Your Device",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),

  # Cancellation alerts causing immediate disabling
  "controlsFailed": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Controls Failed",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "controlsMismatch": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Controls Mismatch",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "canError": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "CAN Error: Check Connections",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "steerUnavailable": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "LKAS Fault: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "brakeUnavailable": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Cruise Fault: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "gasUnavailable": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Gas Fault: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "reverseGear": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Reverse Gear",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "cruiseDisabled": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Cruise Is Off",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "plannerError": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Planner Solution Error",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

  "relayMalfunction": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Harness Malfunction",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),


  # not loud cancellations (user is in control)
  "noTarget": Alert(
      "openpilot Canceled",
      "No close lead car",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),

  "speedTooLow": Alert(
      "openpilot Canceled",
      "Speed too low",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),

  "speedTooHigh": Alert(
      "Speed Too High",
      "Slow down to resume operation",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),

  # Cancellation alerts causing non-entry
  "overheatNoEntry": NoEntryAlert("System overheated"),

  "wrongGearNoEntry": NoEntryAlert("Gear not D"),

  "calibrationInvalidNoEntry": NoEntryAlert("Calibration Invalid: Reposition Device & Recalibrate"),

  "calibrationIncompleteNoEntry": NoEntryAlert("Calibration in Progress"),

  "doorOpenNoEntry": NoEntryAlert("Door open"),

  "seatbeltNotLatchedNoEntry": NoEntryAlert("Seatbelt unlatched"),

  "espDisabledNoEntry": NoEntryAlert("ESP Off"),

  "radarCanErrorNoEntry": NoEntryAlert("Radar Error: Restart the Car"),

  "radarFaultNoEntry": NoEntryAlert("Radar Error: Restart the Car"),

  "posenetInvalidNoEntry": NoEntryAlert("Vision Model Output Uncertain"),

  "controlsFailedNoEntry": NoEntryAlert("Controls Failed"),

  "canErrorNoEntry": NoEntryAlert("CAN Error: Check Connections"),

  "steerUnavailableNoEntry": NoEntryAlert("LKAS Fault: Restart the Car"),

  "brakeUnavailableNoEntry": NoEntryAlert("Cruise Fault: Restart the Car"),

  "gasUnavailableNoEntry": NoEntryAlert("Gas Error: Restart the Car"),

  "reverseGearNoEntry": NoEntryAlert("Reverse Gear"),

  "cruiseDisabledNoEntry": NoEntryAlert("Cruise is Off"),

  "noTargetNoEntry": NoEntryAlert("No Close Lead Car"),

  "plannerErrorNoEntry": NoEntryAlert("Planner Solution Error"),

  "commIssueNoEntry": NoEntryAlert("Communication Issue between Processes",
                                   audible_alert=AudibleAlert.chimeDisengage),

  "radarCommIssueNoEntry": NoEntryAlert("Radar Communication Issue",
                                        audible_alert=AudibleAlert.chimeDisengage),

  "internetConnectivityNeededNoEntry": NoEntryAlert("Please Connect to Internet",
                                                    audible_alert=AudibleAlert.chimeDisengage),

  "lowMemoryNoEntry": NoEntryAlert("Low Memory: Reboot Your Device",
                                   audible_alert=AudibleAlert.chimeDisengage),

  "relayMalfunctionNoEntry": NoEntryAlert("Harness Malfunction"),

  "speedTooHighNoEntry": Alert(
      "Speed Too High",
      "Slow down to engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.chimeError, .4, 2., 3.),

  # permanent alerts
  "steerUnavailablePermanent": Alert(
      "LKAS Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "brakeUnavailablePermanent": Alert(
      "Cruise Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "lowSpeedLockoutPermanent": Alert(
      "Cruise Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "calibrationIncompletePermanent": Alert(
      "Calibration in Progress: ",
      "Drive Above ",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "invalidGiraffeToyotaPermanent": Alert(
      "Unsupported Giraffe Configuration",
      "Visit comma.ai/tg",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "invalidLkasSettingPermanent": Alert(
      "Stock LKAS is turned on",
      "Turn off stock LKAS to engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "internetConnectivityNeededPermanent": Alert(
      "Please connect to Internet",
      "An Update Check Is Required to Engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "communityFeatureDisallowedPermanent": Alert(
      "",
      "Community Feature Detected",
      "Enable Community Features in Developer Settings",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),  # LOW priority to overcome Cruise Error

  "sensorDataInvalidPermanent": Alert(
      "No Data from Device Sensors",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "soundsUnavailablePermanent": Alert(
      "Speaker not found",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "lowMemoryPermanent": Alert(
      "RAM Critically Low",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "carUnrecognizedPermanent": Alert(
      "Dashcam Mode",
      "Car Unrecognized",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "relayMalfunctionPermanent": Alert(
      "Harness Malfunction",
      "Please Check Hardware",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),

  "vehicleModelInvalid": Alert(
      "Vehicle Parameter Identification Failed",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWEST, VisualAlert.steerRequired, AudibleAlert.none, .0, .0, .1),

  "ldwPermanent": Alert(
      "TAKE CONTROL",
      "Lane Departure Detected",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, 1., 2., 3.),
}
