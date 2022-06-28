from cereal import car
from selfdrive.car.chrysler.values import CAR

GearShifter = car.CarState.GearShifter
VisualAlert = car.CarControl.HUDControl.VisualAlert

def create_lkas_hud(packer, lkas_active, hud_alert, hud_count, CS, fingerprint):
  # LKAS_HUD 0x2a6 (678) Controls what lane-keeping icon is displayed.

  color = 1  # default values are for park or neutral in 2017 are 0 0, but trying 1 1 for 2019
  lines = 1
  alerts = 0
  carmodel = 0

  if hud_count < (1 * 4):  # first 3 seconds, 4Hz
    alerts = 1

  # CAR.PACIFICA_2018_HYBRID and CAR.PACIFICA_2019_HYBRID
  # had color = 1 and lines = 1 but trying 2017 hybrid style for now.
  # Lines
  # 03 White Lines
  # 04 grey lines
  # 09 left lane close
  # 0A right lane close
  # 0B Left Lane very close
  # 0C Right Lane very close
  # 0D left cross cross
  # 0E right lane cross

  # Alerts
  # 7 Normal
  # 6 lane departure place hands on wheel

  if CS.out.gearShifter in (GearShifter.drive, GearShifter.reverse, GearShifter.low):
    if lkas_active:
      color = 2  # control active, display green.
      lines = 3
      alerts = 7
    else:
      color = 1  # control off, display white.
      lines = 0
      alerts = 7

  if hud_alert == VisualAlert.ldw:
    color = 4
    lines = 0
    alerts = 6
  elif hud_alert == VisualAlert.steerRequired:
    color = 0
    lines = 0
    alerts = 0
    carmodel = 0xf

  # TODO: what is this? why is it different?
  if fingerprint != CAR.RAM_1500:
    carmodel = CS.lkas_car_model

  values = {
    "LKAS_ICON_COLOR": color,
    "CAR_MODEL": carmodel,  # TODO: look into this
    "LKAS_LANE_LINES": lines,
    "LKAS_ALERTS": alerts,
    "LKAS_Disabled": 0,
  }
  return packer.make_can_msg("DAS_6", 0, values)


def create_lkas_command(packer, apply_steer, moving_fast, frame):
  # LKAS_COMMAND Lane-keeping signal to turn the wheel.
  values = {
    "LKAS_STEERING_TORQUE": apply_steer,
    "LKAS_CONTROL_BIT": 2 if moving_fast else 0,  # 0=IDLE, 2=LKAS
    "COUNTER": frame % 0x10,
  }
  return packer.make_can_msg("LKAS_COMMAND", 0, values)


def create_wheel_buttons(packer, frame, bus, cancel=False):
  values = {
    "ACC_Cancel": cancel,
    "COUNTER": frame % 0x10,
  }
  return packer.make_can_msg("CRUISE_BUTTONS", bus, values)
