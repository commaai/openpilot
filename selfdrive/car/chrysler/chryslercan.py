from cereal import car
from selfdrive.car import make_can_msg
from selfdrive.car.chrysler.values import CAR


GearShifter = car.CarState.GearShifter
VisualAlert = car.CarControl.HUDControl.VisualAlert

def create_lkas_hud(packer, lkas_active, hud_alert, hud_count, CS, fingerprint):
  # LKAS_HUD 0x2a6 (678) Controls what lane-keeping icon is displayed.

  #if hud_alert in (VisualAlert.steerRequired):
  #  if fingerprint in (CAR.RAM_1500, CAR.RAM_2500):
  #    msg = b'\x00\x00\x0F\x00\x00\x00\x00\x00'
  #  else:
  #    msg = b'\x00\x00\x00\x03\x00\x00\x00\x00'
  #  return make_can_msg(0x2a6, msg, 0)
    
  lkasdisabled = CS.lkasdisabled
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

  #Alerts
  #7 Normal
  #6 lane departure place hands on wheel

  if CS.out.gearShifter in (GearShifter.drive, GearShifter.reverse, GearShifter.low):
    if lkas_active:
      color = 2  # control active, display green.
      lines = 3
      alerts = 7
    else:
      color = 1  # control off, display white.
      lines = 0
      alerts = 7
  if CS.lkasdisabled == 1:
    color = 0
    lines = 0
    alerts = 0

  if hud_alert in [VisualAlert.ldw]: #possible use this instead
    color = 4
    lines = 0
    alerts = 6

  if hud_alert in [VisualAlert.steerRequired]: 
    color = 0
    lines = 0
    alerts = 0
    carmodel = 0xf

  if fingerprint in (CAR.RAM_1500, CAR.RAM_2500):
    values = {
      "Auto_High_Beam": CS.autoHighBeamBit,
      "LKAS_ICON_COLOR": color,  # byte 0, last 2 bits
      "CAR_MODEL": carmodel,  # byte 1
      "LKAS_LANE_LINES": lines,  # byte 2, last 4 bits
      "LKAS_ALERTS": alerts,  # byte 3, last 4 bits
      "LKAS_Disabled":lkasdisabled,
    }

  else:
    values = {
      "LKAS_ICON_COLOR": color,  # byte 0, last 2 bits
      "CAR_MODEL": CS.lkas_car_model,  # byte 1
      "LKAS_LANE_LINES": lines,  # byte 2, last 4 bits
      "LKAS_ALERTS": alerts,  # byte 3, last 4 bits 
      "LKAS_Disabled":lkasdisabled,
      }

  

  return packer.make_can_msg("DAS_6", 0, values)


def create_lkas_command(packer, apply_steer, moving_fast, frame):
  # LKAS_COMMAND Lane-keeping signal to turn the wheel.
  values = {
    "LKAS_STEERING_TORQUE": apply_steer,
    "LKAS_CONTROL_BIT": int(moving_fast),
    "COUNTER": frame % 0x10,
  }
  return packer.make_can_msg("LKAS_COMMAND", 0, values)


def create_wheel_buttons(packer, frame, fingerprint, cancel = False, acc_resume = False):
  # Cruise_Control_Buttons Message sent to cancel ACC.
  values = {
    "ACC_Cancel": cancel,
    "COUNTER": frame % 0x10,
    "ACC_Resume": acc_resume,
  }
  if fingerprint in (CAR.RAM_1500, CAR.RAM_2500):
    bus = 2
  else:
    bus = 0 

  return packer.make_can_msg("Cruise_Control_Buttons", bus, values)