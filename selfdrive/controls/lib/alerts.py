from cereal import car, log
from selfdrive.controls.lib.drive_helpers import EventTypes as ET

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
EN = car.CarEvent.EventName

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

enable_alert =  Alert(
                    "",
                    "",
                    AlertStatus.normal, AlertSize.none,
                    Priority.MID, VisualAlert.none, AudibleAlert.chimeEngage, .2, 0., 0.)

disable_alert = Alert(
                    "",
                    "",
                    AlertStatus.normal, AlertSize.none,
                    Priority.MID, VisualAlert.none, AudibleAlert.chimeDisengage, .2, 0., 0.)

ALERTS = {
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

  "debug": Alert(
      "DEBUG ALERT",
      "",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .1, .1, .1),

  "ethicalDilemma": Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Ethical Dilemma Detected",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 1., 3., 3.),

  "steerSaturated": Alert(
      "TAKE CONTROL",
      "Turn Exceeds Steering Limit",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, 1., 2., 3.),

  "ldwPermanent": Alert(
      "TAKE CONTROL",
      "Lane Departure Detected",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, 1., 2., 3.),
}

EVENT_ALERTS = {
  EN.pcmEnable: {ET.ENABLE: enable_alert},
  EN.buttonEnable: {ET.ENABLE: enable_alert},

  EN.pcmDisable: {ET.USER_DISABLE: disable_alert},
  EN.buttonCancel: {ET.USER_DISABLE: disable_alert},

  EN.brakeHold: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Brake Hold Active"),
  },

  EN.parkBrake: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Park Brake Engaged"),
  },

  EN.pedalPressed: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Pedal Pressed During Attempt",
                              visual_alert=VisualAlert.brakePressed),
  },

  EN.wrongCarMode: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Main Switch Off",
                              duration_hud_alert=0.),
  },

  EN.steerTempUnavailable: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Steering Temporarily Unavailable",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),
    ET.NO_ENTRY: NoEntryAlert("Steering Temporarily Unavailable",
                              duration_hud_alert=0.),
  },

  EN.steerTempUnavailableMute: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Steering Temporarily Unavailable",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .2, .2, .2),
  },

  EN.preDriverDistracted: {
    ET.WARNING: Alert(
      "KEEP EYES ON ROAD: Driver Distracted",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EN.promptDriverDistracted: {
    ET.WARNING: Alert(
      "KEEP EYES ON ROAD",
      "Driver Appears Distracted",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),
  },

  EN.driverDistracted: {
    ET.WARNING: Alert(
      "DISENGAGE IMMEDIATELY",
      "Driver Was Distracted",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),
  },

  EN.preDriverUnresponsive: {
    ET.WARNING: Alert(
      "TOUCH STEERING WHEEL: No Face Detected",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EN.promptDriverUnresponsive: {
    ET.WARNING: Alert(
      "TOUCH STEERING WHEEL",
      "Driver Is Unresponsive",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),
  },

  EN.driverUnresponsive: {
    ET.WARNING: Alert(
      "DISENGAGE IMMEDIATELY",
      "Driver Was Unresponsive",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),
  },

  EN.driverMonitorLowAcc: {
    ET.WARNING: Alert(
      "CHECK DRIVER FACE VISIBILITY",
      "Driver Monitor Model Output Uncertain",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .4, 0., 1.),
  },

  EN.manualRestart: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Resume Driving Manually",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EN.resumeRequired: {
    ET.WARNING: Alert(
      "STOPPED",
      "Press Resume to Move",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EN.belowSteerSpeed: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Steer Unavailable Below ",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.none, 0., 0.4, .3),
  },

  EN.preLaneChangeLeft: {
    ET.WARNING: Alert(
      "Steer Left to Start Lane Change",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EN.preLaneChangeRight: {
    ET.WARNING: Alert(
      "Steer Right to Start Lane Change",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },


  EN.laneChange: {
    ET.WARNING: Alert(
      "Changing Lane",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1),
  },


  EN.posenetInvalid: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Vision Model Output Uncertain",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),
    ET.NO_ENTRY: NoEntryAlert("Vision Model Output Uncertain"),
  },


  # Non-entry only alerts

  # TODO: unused?
  #EN.dataNeeded: {
  #  ET.NO_ENTRY: NoEntryAlert("Calibration Needs Data. Upload Drive, Try Again",
  #               duration_hud_alert=0.),
  #},

  EN.outOfSpace: {
    ET.NO_ENTRY: NoEntryAlert("Out of Storage Space",
                              duration_hud_alert=0.),
  },

  EN.sensorDataInvalid: {
    ET.PERMANENT: Alert(
      "No Data from Device Sensors",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("No Data from Device Sensors"),
  },

  EN.soundsUnavailable: {
    ET.PERMANENT: Alert(
      "Speaker not found",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Speaker not found"),
  },

  EN.tooDistracted: {
    ET.NO_ENTRY: NoEntryAlert("Distraction Level Too High"),
  },

  # Cancellation alerts causing soft disabling
  EN.overheat: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "System Overheated",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("System overheated"),
  },

  EN.wrongGear: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Gear not D",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Gear not D"),
  },

  EN.calibrationInvalid: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Calibration Invalid: Reposition Device and Recalibrate",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Calibration Invalid: Reposition Device & Recalibrate"),
  },

  EN.calibrationIncomplete: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Calibration in Progress",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.PERMANENT: Alert(
      "Calibration in Progress: ",
      "Drive Above ",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Calibration in Progress"),
  },

  EN.doorOpen: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Door Open",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Door open"),
  },

  EN.seatbeltNotLatched: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Seatbelt Unlatched",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Seatbelt unlatched"),
  },

  EN.espDisabled: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "ESP Off",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("ESP Off"),
  },

  EN.lowBattery: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Low Battery",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Low Battery"),
  },

  EN.commIssue: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Communication Issue between Processes",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Communication Issue between Processes",
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  EN.radarCommIssue: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Radar Communication Issue",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Radar Communication Issue",
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  EN.radarCanError: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Radar Error: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Radar Error: Restart the Car"),
  },

  EN.radarFault: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Radar Error: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.NO_ENTRY : NoEntryAlert("Radar Error: Restart the Car"),
  },


  EN.lowMemory: {
    ET.SOFT_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Low Memory: Reboot Your Device",
      AlertStatus.critical, AlertSize.full,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, 2., 2.),
    ET.PERMANENT: Alert(
      "RAM Critically Low",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY : NoEntryAlert("Low Memory: Reboot Your Device",
                               audible_alert=AudibleAlert.chimeDisengage),
  },

  # Cancellation alerts causing immediate disabling
  EN.controlsFailed: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Controls Failed",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.NO_ENTRY: NoEntryAlert("Controls Failed"),
  },

  EN.controlsMismatch: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Controls Mismatch",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
  },

  EN.canError: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "CAN Error: Check Connections",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.NO_ENTRY: NoEntryAlert("CAN Error: Check Connections"),
  },

  EN.steerUnavailable: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "LKAS Fault: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.PERMANENT: Alert(
      "LKAS Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("LKAS Fault: Restart the Car"),
  },

  EN.brakeUnavailable: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Cruise Fault: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.PERMANENT: Alert(
      "Cruise Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Cruise Fault: Restart the Car"),
  },

  EN.gasUnavailable: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Gas Fault: Restart the Car",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.NO_ENTRY: NoEntryAlert("Gas Error: Restart the Car"),
  },

  EN.reverseGear: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Reverse Gear",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.NO_ENTRY: NoEntryAlert("Reverse Gear"),
  },

  EN.cruiseDisabled: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Cruise Is Off",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.NO_ENTRY: NoEntryAlert("Cruise is Off"),
  },

  EN.plannerError: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Planner Solution Error",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.NO_ENTRY: NoEntryAlert("Planner Solution Error"),
  },

  EN.relayMalfunction: {
    ET.IMMEDIATE_DISABLE: Alert(
      "TAKE CONTROL IMMEDIATELY",
      "Harness Malfunction",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.PERMANENT: Alert(
      "Harness Malfunction",
      "Please Check Hardware",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Harness Malfunction"),
  },


  # not loud cancellations (user is in control)
  EN.noTarget: {
    ET.IMMEDIATE_DISABLE: Alert(
      "openpilot Canceled",
      "No close lead car",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),
    ET.NO_ENTRY : NoEntryAlert("No Close Lead Car"),
  },

  EN.speedTooLow: {
    ET.IMMEDIATE_DISABLE: Alert(
      "openpilot Canceled",
      "Speed too low",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),
    ET.NO_ENTRY: NoEntryAlert("Speed Too Low"),
  },

  EN.speedTooHigh: {
    ET.IMMEDIATE_DISABLE: Alert(
      "Speed Too High",
      "Slow down to resume operation",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),
    ET.NO_ENTRY: Alert(
      "Speed Too High",
      "Slow down to engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.chimeError, .4, 2., 3.),
  },

  EN.internetConnectivityNeeded: {
    ET.PERMANENT: Alert(
      "Please connect to Internet",
      "An Update Check Is Required to Engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Please Connect to Internet",
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  # permanent alerts

  EN.lowSpeedLockout: {
    ET.PERMANENT: Alert(
      "Cruise Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Cruise Fault: Restart the Car"),
  },

  EN.invalidGiraffeToyota: {
    ET.PERMANENT: Alert(
      "Unsupported Giraffe Configuration",
      "Visit comma.ai/tg",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EN.invalidLkasSetting: {
    ET.PERMANENT: Alert(
      "Stock LKAS is turned on",
      "Turn off stock LKAS to engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EN.communityFeatureDisallowed: {
    ET.PERMANENT: Alert(
      "",
      "Community Feature Detected",
      "Enable Community Features in Developer Settings",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),  # LOW priority to overcome Cruise Error
  },

  EN.carUnrecognized: {
    ET.PERMANENT: Alert(
      "Dashcam Mode",
      "Car Unrecognized",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EN.vehicleModelInvalid: {
    ET.PERMANENT: Alert(
      "Vehicle Parameter Identification Failed",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWEST, VisualAlert.steerRequired, AudibleAlert.none, .0, .0, .1),
  },

}
