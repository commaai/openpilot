# This Python file uses the following encoding: utf-8
# -*- coding: utf-8 -*-
from enum import IntEnum
from typing import Dict, Union, Callable, Any

from cereal import log, car
import cereal.messaging as messaging
from common.realtime import DT_CTRL
from selfdrive.config import Conversions as CV
from selfdrive.locationd.calibrationd import MIN_SPEED_FILTER
from common.i18n import events
_ = events()

AlertSize = log.ControlsState.AlertSize
AlertStatus = log.ControlsState.AlertStatus
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert
EventName = car.CarEvent.EventName

# Alert priorities
class Priority(IntEnum):
  LOWEST = 0
  LOWER = 1
  LOW = 2
  MID = 3
  HIGH = 4
  HIGHEST = 5

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
    self.static_events = []
    self.events_prev = dict.fromkeys(EVENTS.keys(), 0)

  @property
  def names(self):
    return self.events

  def __len__(self):
    return len(self.events)

  def add(self, event_name, static=False):
    if static:
      self.static_events.append(event_name)
    self.events.append(event_name)

  def clear(self):
    self.events_prev = {k: (v+1 if k in self.events else 0) for k, v in self.events_prev.items()}
    self.events = self.static_events.copy()

  def any(self, event_type):
    for e in self.events:
      if event_type in EVENTS.get(e, {}).keys():
        return True
    return False

  def create_alerts(self, event_types, callback_args=None):
    if callback_args is None:
      callback_args = []

    ret = []
    for e in self.events:
      types = EVENTS[e].keys()
      for et in event_types:
        if et in types:
          alert = EVENTS[e][et]
          if not isinstance(alert, Alert):
            alert = alert(*callback_args)

          if DT_CTRL * (self.events_prev[e] + 1) >= alert.creation_delay:
            alert.alert_type = f"{EVENT_NAME[e]}/{et}"
            alert.event_type = et
            ret.append(alert)
    return ret

  def add_from_msg(self, events):
    for e in events:
      self.events.append(e.name.raw)

  def to_msg(self):
    ret = []
    for event_name in self.events:
      event = car.CarEvent.new_message()
      event.name = event_name
      for event_type in EVENTS.get(event_name, {}).keys():
        setattr(event, event_type , True)
      ret.append(event)
    return ret

class Alert:
  def __init__(self,
               alert_text_1: str,
               alert_text_2: str,
               alert_status: log.ControlsState.AlertStatus,
               alert_size: log.ControlsState.AlertSize,
               alert_priority: Priority,
               visual_alert: car.CarControl.HUDControl.VisualAlert,
               audible_alert: car.CarControl.HUDControl.AudibleAlert,
               duration_sound: float,
               duration_hud_alert: float,
               duration_text: float,
               alert_rate: float = 0.,
               creation_delay: float = 0.):

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

    self.alert_rate = alert_rate
    self.creation_delay = creation_delay

    self.start_time = 0.
    self.alert_type = ""
    self.event_type = None

  def __str__(self) -> str:
    return f"{self.alert_text_1}/{self.alert_text_2} {self.alert_priority} {self.visual_alert} {self.audible_alert}"

  def __gt__(self, alert2) -> bool:
    return self.alert_priority > alert2.alert_priority

class NoEntryAlert(Alert):
  def __init__(self, alert_text_2, audible_alert=AudibleAlert.chimeError,
               visual_alert=VisualAlert.none, duration_hud_alert=2.):
    super().__init__(_("openpilot Unavailable"), alert_text_2, AlertStatus.normal,
                     AlertSize.mid, Priority.LOW, visual_alert,
                     audible_alert, .4, duration_hud_alert, 3.)


class SoftDisableAlert(Alert):
  def __init__(self, alert_text_2):
    super().__init__(_("TAKE CONTROL IMMEDIATELY"), alert_text_2,
                     AlertStatus.critical, AlertSize.full,
                     Priority.MID, VisualAlert.steerRequired,
                     AudibleAlert.chimeWarningRepeat, .1, 2., 2.),


class ImmediateDisableAlert(Alert):
  def __init__(self, alert_text_2, alert_text_1=_("TAKE CONTROL IMMEDIATELY")):
    super().__init__(alert_text_1, alert_text_2,
                     AlertStatus.critical, AlertSize.full,
                     Priority.HIGHEST, VisualAlert.steerRequired,
                     AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),

class EngagementAlert(Alert):
  def __init__(self, audible_alert=True):
    super().__init__("", "",
                     AlertStatus.normal, AlertSize.none,
                     Priority.MID, VisualAlert.none,
                     audible_alert, .2, 0., 0.),

class NormalPermanentAlert(Alert):
  def __init__(self, alert_text_1: str, alert_text_2: str, duration_text: float = 0.2):
    super().__init__(alert_text_1, alert_text_2,
                     AlertStatus.normal, AlertSize.mid if len(alert_text_2) else AlertSize.small,
                     Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., duration_text),

# ********** alert callback functions **********

def below_steer_speed_alert(CP: car.CarParams, sm: messaging.SubMaster, metric: bool) -> Alert:
  speed = int(round(CP.minSteerSpeed * (CV.MS_TO_KPH if metric else CV.MS_TO_MPH)))
  unit = "km/h" if metric else "mph"
  return Alert(
    _("TAKE CONTROL"),
    _("Steer Unavailable Below %(speed)d %(unit)s") % ({"speed": speed, "unit": unit}),
    AlertStatus.userPrompt, AlertSize.mid,
    Priority.MID, VisualAlert.steerRequired, AudibleAlert.none, 0., 0.4, .3)

def calibration_incomplete_alert(CP: car.CarParams, sm: messaging.SubMaster, metric: bool) -> Alert:
  speed = int(MIN_SPEED_FILTER * (CV.MS_TO_KPH if metric else CV.MS_TO_MPH))
  unit = "km/h" if metric else "mph"
  return Alert(
    _("Calibration in Progress: %d%%") % sm['liveCalibration'].calPerc,
    _("Drive Above %(speed)d %(unit)s") % ({"speed": speed, "unit": unit}),
    AlertStatus.normal, AlertSize.mid,
    Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0., 0., .2)

def no_gps_alert(CP: car.CarParams, sm: messaging.SubMaster, metric: bool) -> Alert:
  gps_integrated = sm['pandaState'].pandaType in [log.PandaState.PandaType.uno, log.PandaState.PandaType.dos]
  return Alert(
    _("Poor GPS reception"),
    _("If sky is visible, contact support") if gps_integrated else _("Check GPS antenna placement"),
    AlertStatus.normal, AlertSize.mid,
    Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2, creation_delay=300.)

def wrong_car_mode_alert(CP: car.CarParams, sm: messaging.SubMaster, metric: bool) -> Alert:
  text = _("Cruise Mode Disabled")
  if CP.carName == "honda":
    text = _("Main Switch Off")
  return NoEntryAlert(text, duration_hud_alert=0.)

def startup_fuzzy_fingerprint_alert(CP: car.CarParams, sm: messaging.SubMaster, metric: bool) -> Alert:
  return Alert(
    "WARNING: No Exact Match on Car Model",
    f"Closest Match: {CP.carFingerprint.title()[:40]}",
    AlertStatus.userPrompt, AlertSize.mid,
    Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.)

EVENTS: Dict[int, Dict[str, Union[Alert, Callable[[Any, messaging.SubMaster, bool], Alert]]]] = {
  # ********** events with no alerts **********

  # ********** events only containing alerts displayed in all states **********

  EventName.joystickDebug: {
    ET.PERMANENT: Alert(
      _("DEBUG ALERT"),
      "",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .1, .1, .1),
  },

  EventName.controlsInitializing: {
    ET.NO_ENTRY: NoEntryAlert("Controls Initializing"),
  },

  EventName.startup: {
    ET.PERMANENT: Alert(
      _("Be ready to take over at any time"),
      _("Always keep hands on wheel and eyes on road"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),
  },

  EventName.startupMaster: {
    ET.PERMANENT: Alert(
      _("WARNING: This branch is not tested"),
      _("Always keep hands on wheel and eyes on road"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),
  },

  EventName.startupNoControl: {
    ET.PERMANENT: Alert(
      _("Dashcam mode"),
      _("Always keep hands on wheel and eyes on road"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),
  },

  EventName.startupNoCar: {
    ET.PERMANENT: Alert(
      _("Dashcam mode for unsupported car"),
      _("Always keep hands on wheel and eyes on road"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),
  },

  EventName.startupFuzzyFingerprint: {
    ET.PERMANENT: startup_fuzzy_fingerprint_alert,
  },

  EventName.startupNoFw: {
    ET.PERMANENT: Alert(
      "Car Unrecognized",
      "Check All Connections",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., 15.),
  },

  EventName.dashcamMode: {
    ET.PERMANENT: Alert(
      "Dashcam Mode",
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.invalidLkasSetting: {
    ET.PERMANENT: Alert(
      _("Stock LKAS is turned on"),
      _("Turn off stock LKAS to engage"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.communityFeatureDisallowed: {
    # LOW priority to overcome Cruise Error
    ET.PERMANENT: Alert(
      _("openpilot Not Available"),
      _("Enable Community Features in Settings to Engage"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.carUnrecognized: {
    ET.PERMANENT: Alert(
      _("Dashcam Mode"),
      _("Car Unrecognized"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.stockAeb: {
    ET.PERMANENT: Alert(
      _("BRAKE!"),
      _("Stock AEB: Risk of Collision"),
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.none, 1., 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Stock AEB: Risk of Collision"),
  },

  EventName.stockFcw: {
    ET.PERMANENT: Alert(
      _("BRAKE!"),
      _("Stock FCW: Risk of Collision"),
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.none, 1., 2., 2.),
    ET.NO_ENTRY: NoEntryAlert("Stock FCW: Risk of Collision"),
  },

  EventName.fcw: {
    ET.PERMANENT: Alert(
      _("BRAKE!"),
      _("Risk of Collision"),
      AlertStatus.critical, AlertSize.full,
      Priority.HIGHEST, VisualAlert.fcw, AudibleAlert.chimeWarningRepeat, 1., 2., 2.),
  },

  EventName.ldw: {
    ET.PERMANENT: Alert(
      _("TAKE CONTROL"),
      _("Lane Departure Detected"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, 1., 2., 3.),
  },

  # ********** events only containing alerts that display while engaged **********

  EventName.gasPressed: {
    ET.PRE_ENABLE: Alert(
      _("openpilot will not brake while gas pressed"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, .0, .0, .1, creation_delay=1.),
  },

  EventName.vehicleModelInvalid: {
    ET.NO_ENTRY: NoEntryAlert("Vehicle Parameter Identification Failed"),
    ET.SOFT_DISABLE: SoftDisableAlert("Vehicle Parameter Identification Failed"),
    ET.WARNING: Alert(
      _("Vehicle Parameter Identification Failed"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWEST, VisualAlert.steerRequired, AudibleAlert.none, .0, .0, .1),
  },

  EventName.steerTempUnavailableUserOverride: {
    ET.WARNING: Alert(
      _("Steering Temporarily Unavailable"),
      "",
      AlertStatus.userPrompt, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, 1., 1., 1.),
  },

  EventName.preDriverDistracted: {
    ET.WARNING: Alert(
      _("KEEP EYES ON ROAD: Driver Distracted"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1),
  },

  EventName.promptDriverDistracted: {
    ET.WARNING: Alert(
      _("KEEP EYES ON ROAD"),
      _("Driver Distracted"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),
  },

  EventName.driverDistracted: {
    ET.WARNING: Alert(
      _("DISENGAGE IMMEDIATELY"),
      _("Driver Distracted"),
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),
  },

  EventName.preDriverUnresponsive: {
    ET.WARNING: Alert(
      _("TOUCH STEERING WHEEL: No Face Detected"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EventName.promptDriverUnresponsive: {
    ET.WARNING: Alert(
      _("TOUCH STEERING WHEEL"),
      _("Driver Unresponsive"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.chimeWarning2Repeat, .1, .1, .1),
  },

  EventName.driverUnresponsive: {
    ET.WARNING: Alert(
      _("DISENGAGE IMMEDIATELY"),
      _("Driver Unresponsive"),
      AlertStatus.critical, AlertSize.full,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, .1, .1, .1),
  },

  EventName.driverMonitorLowAcc: {
    ET.WARNING: Alert(
      _("CHECK DRIVER FACE VISIBILITY"),
      _("Driver Monitoring Uncertain"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .4, 0., 1.5),
  },

  EventName.manualRestart: {
    ET.WARNING: Alert(
      _("TAKE CONTROL"),
      _("Resume Driving Manually"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.resumeRequired: {
    ET.WARNING: Alert(
      _("STOPPED"),
      _("Press Resume to Move"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
  },

  EventName.belowSteerSpeed: {
    ET.WARNING: below_steer_speed_alert,
  },

  EventName.preLaneChangeLeft: {
    ET.WARNING: Alert(
      _("Steer Left to Start Lane Change"),
      _("Monitor Other Vehicles"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EventName.preLaneChangeRight: {
    ET.WARNING: Alert(
      _("Steer Right to Start Lane Change"),
      _("Monitor Other Vehicles"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1, alert_rate=0.75),
  },

  EventName.laneChangeBlocked: {
    ET.WARNING: Alert(
      _("Car Detected in Blindspot"),
      _("Monitor Other Vehicles"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, .1, .1, .1),
  },

  EventName.laneChange: {
    ET.WARNING: Alert(
      _("Changing Lane"),
      _("Monitor Other Vehicles"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.none, .0, .1, .1),
  },

  EventName.steerSaturated: {
    ET.WARNING: Alert(
      _("TAKE CONTROL"),
      _("Turn Exceeds Steering Limit"),
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimePrompt, 1., 1., 1.),
  },

  EventName.fanMalfunction: {
    ET.PERMANENT: NormalPermanentAlert(_("Fan Malfunction"), _("Contact Support")),
  },

  EventName.cameraMalfunction: {
    ET.PERMANENT: NormalPermanentAlert(_("Camera Malfunction"), _("Contact Support")),
  },

  EventName.gpsMalfunction: {
    ET.PERMANENT: NormalPermanentAlert(_("GPS Malfunction"), _("Contact Support")),
  },

  EventName.localizerMalfunction: {
    ET.PERMANENT: NormalPermanentAlert("Localizer unstable", "Contact Support"),
  },

  # ********** events that affect controls state transitions **********

  EventName.pcmEnable: {
    ET.ENABLE: EngagementAlert(AudibleAlert.chimeEngage),
  },

  EventName.buttonEnable: {
    ET.ENABLE: EngagementAlert(AudibleAlert.chimeEngage),
  },

  EventName.pcmDisable: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.chimeDisengage),
  },

  EventName.buttonCancel: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.chimeDisengage),
  },

  EventName.brakeHold: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.chimeDisengage),
    ET.NO_ENTRY: NoEntryAlert(_("Brake Hold Active")),
  },

  EventName.parkBrake: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.chimeDisengage),
    ET.NO_ENTRY: NoEntryAlert(_("Park Brake Engaged")),
  },

  EventName.pedalPressed: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.chimeDisengage),
    ET.NO_ENTRY: NoEntryAlert(_("Pedal Pressed During Attempt"),
                              visual_alert=VisualAlert.brakePressed),
  },

  EventName.wrongCarMode: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.chimeDisengage),
    ET.NO_ENTRY: wrong_car_mode_alert,
  },

  EventName.wrongCruiseMode: {
    ET.USER_DISABLE: EngagementAlert(AudibleAlert.chimeDisengage),
    ET.NO_ENTRY: NoEntryAlert(_("Enable Adaptive Cruise")),
  },

  EventName.steerTempUnavailable: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Steering Temporarily Unavailable")),
    ET.NO_ENTRY: NoEntryAlert(_("Steering Temporarily Unavailable"),
                              duration_hud_alert=0.),
  },

  EventName.outOfSpace: {
    ET.PERMANENT: Alert(
      _("Out of Storage"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert(_("Out of Storage Space"),
                              duration_hud_alert=0.),
  },

  EventName.belowEngageSpeed: {
    ET.NO_ENTRY: NoEntryAlert(_("Speed Too Low")),
  },

  EventName.sensorDataInvalid: {
    ET.PERMANENT: Alert(
      _("No Data from Device Sensors"),
      _("Reboot your Device"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2, creation_delay=1.),
    ET.NO_ENTRY: NoEntryAlert(_("No Data from Device Sensors")),
  },

  EventName.noGps: {
    ET.PERMANENT: no_gps_alert,
  },

  EventName.soundsUnavailable: {
    ET.PERMANENT: NormalPermanentAlert(_("Speaker not found"), _("Reboot your Device")),
    ET.NO_ENTRY: NoEntryAlert(_("Speaker not found")),
  },

  EventName.tooDistracted: {
    ET.NO_ENTRY: NoEntryAlert(_("Distraction Level Too High")),
  },

  EventName.overheat: {
    ET.PERMANENT: Alert(
      _("System Overheated"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.SOFT_DISABLE: SoftDisableAlert(_("System Overheated")),
    ET.NO_ENTRY: NoEntryAlert(_("System Overheated")),
  },

  EventName.wrongGear: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Gear not D")),
    ET.NO_ENTRY: NoEntryAlert(_("Gear not D")),
  },

  EventName.calibrationInvalid: {
    ET.PERMANENT: NormalPermanentAlert(_("Calibration Invalid"), _("Remount Device and Recalibrate")),
    ET.SOFT_DISABLE: SoftDisableAlert(_("Calibration Invalid: Remount Device & Recalibrate")),
    ET.NO_ENTRY: NoEntryAlert(_("Calibration Invalid: Remount Device & Recalibrate")),
  },

  EventName.calibrationIncomplete: {
    ET.PERMANENT: calibration_incomplete_alert,
    ET.SOFT_DISABLE: SoftDisableAlert(_("Calibration in Progress")),
    ET.NO_ENTRY: NoEntryAlert(_("Calibration in Progress")),
  },

  EventName.doorOpen: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Door Open")),
    ET.NO_ENTRY: NoEntryAlert(_("Door Open")),
  },

  EventName.seatbeltNotLatched: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Seatbelt Unlatched")),
    ET.NO_ENTRY: NoEntryAlert(_("Seatbelt Unlatched")),
  },

  EventName.espDisabled: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("ESP Off")),
    ET.NO_ENTRY: NoEntryAlert(_("ESP Off")),
  },

  EventName.lowBattery: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Low Battery")),
    ET.NO_ENTRY: NoEntryAlert(_("Low Battery")),
  },

  EventName.commIssue: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Communication Issue between Processes")),
    ET.NO_ENTRY: NoEntryAlert(_("Communication Issue between Processes"),
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  EventName.processNotRunning: {
    ET.NO_ENTRY: NoEntryAlert(_("System Malfunction: Reboot Your Device"),
                              audible_alert=AudibleAlert.chimeDisengage),
  },

  EventName.radarFault: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Radar Error: Restart the Car")),
    ET.NO_ENTRY : NoEntryAlert(_("Radar Error: Restart the Car")),
  },

  EventName.modeldLagging: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Driving model lagging")),
    ET.NO_ENTRY : NoEntryAlert(_("Driving model lagging")),
  },

  EventName.posenetInvalid: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Model Output Uncertain")),
    ET.NO_ENTRY: NoEntryAlert(_("Model Output Uncertain")),
  },

  EventName.deviceFalling: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Device Fell Off Mount")),
    ET.NO_ENTRY: NoEntryAlert(_("Device Fell Off Mount")),
  },

  EventName.lowMemory: {
    ET.SOFT_DISABLE: SoftDisableAlert(_("Low Memory: Reboot Your Device")),
    ET.PERMANENT: NormalPermanentAlert(_("Low Memory"), _("Reboot your Device")),
    ET.NO_ENTRY : NoEntryAlert(_("Low Memory: Reboot Your Device"),
                               audible_alert=AudibleAlert.chimeDisengage),
  },

  EventName.accFaulted: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("Cruise Faulted")),
    ET.PERMANENT: NormalPermanentAlert(_("Cruise Faulted"), ""),
    ET.NO_ENTRY: NoEntryAlert(_("Cruise Faulted")),
  },

  EventName.controlsMismatch: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("Controls Mismatch")),
  },

  EventName.roadCameraError: {
    ET.PERMANENT: NormalPermanentAlert("Road Camera Error", "",
                                       duration_text=10.),
  },

  EventName.driverCameraError: {
    ET.PERMANENT: NormalPermanentAlert("Driver Camera Error", "",
                                       duration_text=10.),
  },

  EventName.wideRoadCameraError: {
    ET.PERMANENT: NormalPermanentAlert("Wide Road Camera Error", "",
                                       duration_text=10.),
  },

  EventName.usbError: {
    ET.SOFT_DISABLE: SoftDisableAlert("USB Error: Reboot Your Device"),
    ET.PERMANENT: NormalPermanentAlert("USB Error: Reboot Your Device", ""),
    ET.NO_ENTRY: NoEntryAlert("USB Error: Reboot Your Device"),
  },

  EventName.canError: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("CAN Error: Check Connections")),
    ET.PERMANENT: Alert(
      _("CAN Error: Check Connections"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, 0., 0., .2, creation_delay=1.),
    ET.NO_ENTRY: NoEntryAlert(_("CAN Error: Check Connections")),
  },

  EventName.steerUnavailable: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("LKAS Fault: Restart the Car")),
    ET.PERMANENT: Alert(
      _("LKAS Fault: Restart the car to engage"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert(_("LKAS Fault: Restart the Car")),
  },

  EventName.brakeUnavailable: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("Cruise Fault: Restart the Car")),
    ET.PERMANENT: Alert(
      _("Cruise Fault: Restart the car to engage"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert(_("Cruise Fault: Restart the Car")),
  },

  EventName.reverseGear: {
    ET.PERMANENT: Alert(
      _("Reverse\nGear"),
      "",
      AlertStatus.normal, AlertSize.full,
      Priority.LOWEST, VisualAlert.none, AudibleAlert.none, 0., 0., .2, creation_delay=0.5),
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("Reverse Gear")),
    ET.NO_ENTRY: NoEntryAlert(_("Reverse Gear")),
  },

  EventName.cruiseDisabled: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("Cruise Is Off")),
  },

  EventName.plannerError: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("Planner Solution Error")),
    ET.NO_ENTRY: NoEntryAlert(_("Planner Solution Error")),
  },

  EventName.relayMalfunction: {
    ET.IMMEDIATE_DISABLE: ImmediateDisableAlert(_("Harness Malfunction")),
    ET.PERMANENT: NormalPermanentAlert(_("Harness Malfunction"), _("Check Hardware")),
    ET.NO_ENTRY: NoEntryAlert(_("Harness Malfunction")),
  },

  EventName.noTarget: {
    ET.IMMEDIATE_DISABLE: Alert(
      _("openpilot Canceled"),
      _("No close lead car"),
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),
    ET.NO_ENTRY : NoEntryAlert(_("No Close Lead Car")),
  },

  EventName.speedTooLow: {
    ET.IMMEDIATE_DISABLE: Alert(
      _("openpilot Canceled"),
      _("Speed too low"),
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.none, AudibleAlert.chimeDisengage, .4, 2., 3.),
  },

  EventName.speedTooHigh: {
    ET.WARNING: Alert(
      _("Speed Too High"),
      _("Model uncertain at this speed"),
      AlertStatus.normal, AlertSize.mid,
      Priority.HIGH, VisualAlert.steerRequired, AudibleAlert.chimeWarningRepeat, 2.2, 3., 4.),
    ET.NO_ENTRY: Alert(
      _("Speed Too High"),
      _("Slow down to engage"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.none, AudibleAlert.chimeError, .4, 2., 3.),
  },

  EventName.lowSpeedLockout: {
    ET.PERMANENT: Alert(
      _("Cruise Fault: Restart the car to engage"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOWER, VisualAlert.none, AudibleAlert.none, 0., 0., .2),
    ET.NO_ENTRY: NoEntryAlert(_("Cruise Fault: Restart the Car")),
  },

  # dp
  EventName.preLaneChangeLeftALC: {
    ET.WARNING: Alert(
      _("Left ALC will start in 3s"),
      _("Monitor Other Vehicles"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning2, .1, .1, .1, alert_rate=0.75),
  },

  EventName.preLaneChangeRightALC: {
    ET.WARNING: Alert(
      _("Right ALC will start in 3s"),
      _("Monitor Other Vehicles"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning2, .1, .1, .1, alert_rate=0.75),
  },

  EventName.manualSteeringRequired: {
    ET.WARNING: Alert(
      _("STEERING REQUIRED: Lane Keeping OFF"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .0, .1, .1, alert_rate=0.25),
  },

  EventName.manualSteeringRequiredBlinkersOn: {
    ET.WARNING: Alert(
      _("STEERING REQUIRED: Blinkers ON"),
      "",
      AlertStatus.normal, AlertSize.small,
      Priority.LOW, VisualAlert.none, AudibleAlert.none, .0, .1, .1, alert_rate=0.25),
  },

  # timebomb
  EventName.timebombWarn: {
    ET.WARNING: Alert(
      _("WARNING"),
      _("Grab wheel to start bypass"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),
  },

  EventName.timebombBypassing: {
    ET.WARNING: Alert(
      _("BYPASSING"),
      _("HOLD WHEEL"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, .4, 2., 3.),
  },

  EventName.timebombBypassed: {
    ET.WARNING: Alert(
      _("Bypassed!"),
      _("Release wheel when ready"),
      AlertStatus.normal, AlertSize.mid,
      Priority.LOW, VisualAlert.steerRequired, AudibleAlert.chimeWarning1, 3., 2., 3.),
  },
}
