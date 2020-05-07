from cereal import log, car
from selfdrive.controls.lib.alerts import Alert, Priority

AlertSize = log.ControlsState.AlertSize
AlertStatus = log.ControlsState.AlertStatus
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert
EventName = car.CarEvent.EventName

# Event types
class ET:
  ENABLE = 'enable'
  PRE_ENABLE = 'preEnable'
  NO_ENTRY = 'noEntry'
  WARNING = 'warning'
  USER_DISABLE = 'userDisable'
  SOFT_DISABLE = 'softDisable'
  IMMEDIATE_DISABLE = 'immediateDisable'
  PERMANENT = 'permanent'

# get event name from enum
EVENT_NAME = {v: k for k, v in EventName.schema.enumerants.items()}

class Events:
  def __init__(self):
    self.events = []

  @property
  def names(self):
    return self.events

  def __len__(self):
    return len(self.events)

  def add(self, event_name):
    self.events.append(event_name)

  def any(self, event_types):
    for e in self.events:
      et = EVENTS.get(e, {}).keys()
      for t in event_types:
        if t in et:
          return True
    return False

  def get_events(self, event_types):
    ret = []
    for e in self.events:
      et = EVENTS.get(e, {}).keys()
      for t in event_types:
        if t in et:
          ret.append(e)
    return ret

  def add_from_msg(self, events):
    for e in events:
      self.events.append(e.name.raw)

  def to_msg(self):
    ret = []
    for event_name in self.events:
      event = car.CarEvent.new_message()
      event.name = event_name
      for event_type in EVENTS[event_name].keys():
        setattr(event, event_type , True)
      ret.append(event)
    return ret

class NoEntryAlert(Alert):
  def __init__(self, alert_text_2, audible_alert=AudibleAlert.chimeError,
               visual_alert=VisualAlert.none, duration_hud_alert=2.):
    super().__init__("openpilot Unavailable", alert_text_2, AlertStatus.normal,
                     AlertSize.mid, Priority.LOW, visual_alert,
                     audible_alert, .4, duration_hud_alert, 3.)


class SoftDisableAlert(Alert):
  def __init__(self, alert_text_2):
    super().__init__("TAKE CONTROL IMMEDIATELY", alert_text_2,
                     AlertStatus.critical, AlertSize.full,
                     Priority.MID, VisualAlert.steerRequired,
                     AudibleAlert.chimeWarningRepeat, .1, 2., 2.),


class ImmediateDisableAlert(Alert):
  def __init__(self, alert_text_2, alert_text_1="TAKE CONTROL IMMEDIATELY"):
    super().__init__(alert_text_1, alert_text_2,
                     AlertStatus.critical, AlertSize.full,
                     Priority.HIGHEST, VisualAlert.steerRequired,
                     AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),


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

EVENTS = {
  # ********** events with no alerts **********

  EventName.gasPressed: {ET.PRE_ENABLE: None},

  # ********** events with one or more alerts **********

  EventName.pcmEnable: {ET.ENABLE: enable_alert},
  EventName.buttonEnable: {ET.ENABLE: enable_alert},

  EventName.pcmDisable: {ET.USER_DISABLE: disable_alert},
  EventName.buttonCancel: {ET.USER_DISABLE: disable_alert},

  EventName.brakeHold: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Brake Hold Active"),
  },

  EventName.stockFcw: {
    ET.WARNING: Alert(
      "BRAKE!",
      "Risk of Collision",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.none, 1., 2., 2.),
  },

  EventName.parkBrake: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Park Brake Engaged"),
  },

  EventName.pedalPressed: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Pedal Pressed During Attempt",
                              visual_alert=VisualAlert.brakePressed),
  },

  EventName.wrongCarMode: {
    ET.USER_DISABLE: disable_alert,
    ET.NO_ENTRY: NoEntryAlert("Main Switch Off",
                              duration_hud_alert=0.),
  },

  EventName.steerTempUnavailable: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Steering Temporarily Unavailable",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),
    ET.NO_ENTRY: NoEntryAlert("Steering Temporarily Unavailable",
                              duration_hud_alert=0.),
  },

  EventName.steerTempUnavailableMute: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Steering Temporarily Unavailable",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .2, .2, .2),
  },

  EventName.preDriverDistracted: {
    ET.WARNING: Alert(
      "KEEP EYES ON ROAD: Driver Distracted",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EventName.promptDriverDistracted: {
    ET.WARNING: Alert(
      "KEEP EYES ON ROAD",
      "Driver Appears Distracted",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),
  },

  EventName.driverDistracted: {
    ET.WARNING: Alert(
      "DISEventName.AGE IMMEDIATELY",
      "Driver Was Distracted",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),
  },

  EventName.preDriverUnresponsive: {
    ET.WARNING: Alert(
      "TOUCH STEERING WHEEL: No Face Detected",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EventName.promptDriverUnresponsive: {
    ET.WARNING: Alert(
      "TOUCH STEERING WHEEL",
      "Driver Is Unresponsive",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),
  },

  EventName.driverUnresponsive: {
    ET.WARNING: Alert(
      "DISEventName.AGE IMMEDIATELY",
      "Driver Was Unresponsive",
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),
  },

  EventName.driverMonitorLowAcc: {
    ET.WARNING: Alert(
      "CHECK DRIVER FACE VISIBILITY",
      "Driver Monitor Model Output Uncertain",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .4, 0., 1.),
  },

  EventName.manualRestart: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Resume Driving Manually",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.resumeRequired: {
    ET.WARNING: Alert(
      "STOPPED",
      "Press Resume to Move",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.belowSteerSpeed: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Steer Unavailable Below ",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.none, 0., 0.4, .3),
  },

  EventName.preLaneChangeLeft: {
    ET.WARNING: Alert(
      "Steer Left to Start Lane Change",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EventName.preLaneChangeRight: {
    ET.WARNING: Alert(
      "Steer Right to Start Lane Change",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EventName.laneChange: {
    ET.WARNING: Alert(
      "Changing Lane",
      "Monitor Other Vehicles",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1),
  },

  EventName.posenetInvalid: {
    ET.WARNING: Alert(
      "TAKE CONTROL",
      "Vision Model Output Uncertain",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),
    ET.NO_ENTRY: NoEntryAlert("Vision Model Output Uncertain"),
  },

  EventName.outOfSpace: {
    ET.NO_ENTRY: NoEntryAlert("Out of Storage Space",
                              duration_hud_alert=0.),
  },

  EventName.sensorDataInvalid: {
    ET.PERMANENT: Alert(
      "No Data from Device Sensors",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("No Data from Device Sensors"),
  },

  EventName.soundsUnavailable: {
    ET.PERMANENT: Alert(
      "Speaker not found",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Speaker not found"),
  },

  EventName.tooDistracted: {
    ET.NO_ENTRY: NoEntryAlert("Distraction Level Too High"),
  },

  EventName.overheat: {
    ET.SOFT_DISABLE: SoftDisableAlert("System Overheated"),
    ET.NO_ENTRY: NoEntryAlert("System overheated"),
  },

  EventName.wrongGear: {
    ET.SOFT_DISABLE: SoftDisableAlert("Gear not D"),
    ET.NO_ENTRY: NoEntryAlert("Gear not D"),
  },

  EventName.calibrationInvalid: {
    ET.SOFT_DISABLE: SoftDisableAlert("Calibration Invalid: Reposition Device and Recalibrate"),
    ET.NO_ENTRY: NoEntryAlert("Calibration Invalid: Reposition Device & Recalibrate"),
  },

  EventName.calibrationIncomplete: {
    ET.SOFT_DISABLE: SoftDisableAlert("Calibration in Progress"),
    ET.PERMANENT: Alert(
      "Calibration in Progress: ",
      "Drive Above ",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Calibration in Progress"),
  },

  EventName.doorOpen: {
    ET.SOFT_DISABLE: SoftDisableAlert("Door Open"),
    ET.NO_ENTRY: NoEntryAlert("Door open"),
  },

  EventName.seatbeltNotLatched: {
    ET.SOFT_DISABLE: SoftDisableAlert("Seatbelt Unlatched"),
    ET.NO_ENTRY: NoEntryAlert("Seatbelt unlatched"),
  },

  EventName.espDisabled: {
    ET.SOFT_DISABLE: SoftDisableAlert("ESP Off"),
    ET.NO_ENTRY: NoEntryAlert("ESP Off"),
  },

  EventName.lowBattery: {
    ET.SOFT_DISABLE: SoftDisableAlert("Low Battery"),
    ET.NO_ENTRY: NoEntryAlert("Low Battery"),
  },

  EventName.commIssue: {
    ET.SOFT_DISABLE: SoftDisableAlert("Communication Issue between Processes"),
    ET.NO_ENTRY: NoEntryAlert("Communication Issue between Processes",
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  EventName.radarCommIssue: {
    ET.SOFT_DISABLE: SoftDisableAlert("Radar Communication Issue"),
    ET.NO_ENTRY: NoEntryAlert("Radar Communication Issue",
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  EventName.radarCanError: {
    ET.SOFT_DISABLE: SoftDisableAlert("Radar Error: Restart the Car"),
    ET.NO_ENTRY: NoEntryAlert("Radar Error: Restart the Car"),
  },

  EventName.radarFault: {
    ET.SOFT_DISABLE: SoftDisableAlert("Radar Error: Restart the Car"),
    ET.NO_ENTRY : NoEntryAlert("Radar Error: Restart the Car"),
  },

  EventName.lowMemory: {
    ET.SOFT_DISABLE: SoftDisableAlert("Low Memory: Reboot Your Device"),
    ET.PERMANENT: Alert(
      "RAM Critically Low",
      "Reboot your Device",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY : NoEntryAlert("Low Memory: Reboot Your Device",
                               audible_alert=AudibleAlert.chimeDisengage),
  },

  EventName.controlsFailed: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Controls Failed"),
    ET.NO_ENTRY: NoEntryAlert("Controls Failed"),
  },

  EventName.controlsMismatch: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Controls Mismatch"),
  },

  EventName.canError: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("CAN Error: Check Connections"),
    ET.NO_ENTRY: NoEntryAlert("CAN Error: Check Connections"),
  },

  EventName.steerUnavailable: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("LKAS Fault: Restart the Car"),
    ET.PERMANENT: Alert(
      "LKAS Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("LKAS Fault: Restart the Car"),
  },

  EventName.brakeUnavailable: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Cruise Fault: Restart the Car"),
    ET.PERMANENT: Alert(
      "Cruise Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Cruise Fault: Restart the Car"),
  },

  EventName.gasUnavailable: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Gas Fault: Restart the Car"),
    ET.NO_ENTRY: NoEntryAlert("Gas Error: Restart the Car"),
  },

  EventName.reverseGear: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Reverse Gear"),
    ET.NO_ENTRY: NoEntryAlert("Reverse Gear"),
  },

  EventName.cruiseDisabled: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Cruise Is Off"),
  },

  EventName.plannerError: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Planner Solution Error"),
    ET.NO_ENTRY: NoEntryAlert("Planner Solution Error"),
  },

  EventName.relayMalfunction: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert("Harness Malfunction"),
    ET.PERMANENT: Alert(
      "Harness Malfunction",
      "Please Check Hardware",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Harness Malfunction"),
  },

  EventName.noTarget: {
    ET.IMMEDIATE_DISABLE: Alert(
      "openpilot Canceled",
      "No close lead car",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),
    ET.NO_ENTRY : NoEntryAlert("No Close Lead Car"),
  },

  EventName.speedTooLow: {
    ET.IMMEDIATE_DISABLE: Alert(
      "openpilot Canceled",
      "Speed too low",
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),
    ET.NO_ENTRY: NoEntryAlert("Speed Too Low"),
  },

  EventName.speedTooHigh: {
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

  EventName.internetConnectivityNeeded: {
    ET.PERMANENT: Alert(
      "Please connect to Internet",
      "An Update Check Is Required to Engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Please Connect to Internet",
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  EventName.lowSpeedLockout: {
    ET.PERMANENT: Alert(
      "Cruise Fault: Restart the car to engage",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert("Cruise Fault: Restart the Car"),
  },

  EventName.invalidGiraffeToyota: {
    ET.PERMANENT: Alert(
      "Unsupported Giraffe Configuration",
      "Visit comma.ai/tg",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.invalidLkasSetting: {
    ET.PERMANENT: Alert(
      "Stock LKAS is turned on",
      "Turn off stock LKAS to engage",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.communityFeatureDisallowed: {
    # LOW priority to overcome Cruise Error
    ET.PERMANENT: Alert(
      "",
      "Community Feature Detected",
      "Enable Community Features in Developer Settings",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.carUnrecognized: {
    ET.PERMANENT: Alert(
      "Dashcam Mode",
      "Car Unrecognized",
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.vehicleModelInvalid: {
    ET.WARNING: Alert(
      "Vehicle Parameter Identification Failed",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWEST, VisualAlert.steerRequired, AudibleAlert.none, .0, .0, .1),
  },

}
