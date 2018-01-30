import struct

import common.numpy_fast as np
from selfdrive.config import Conversions as CV
from common.fingerprints import HONDA as CAR


# *** Honda specific ***
def can_cksum(mm):
  s = 0
  for c in mm:
    c = ord(c)
    s += (c>>4)
    s += c & 0xF
  s = 8-s
  s %= 0x10
  return s


def fix(msg, addr):
  msg2 = msg[0:-1] + chr(ord(msg[-1]) | can_cksum(struct.pack("I", addr)+msg))
  return msg2


def make_can_msg(addr, dat, idx, alt):
  if idx is not None:
    dat += chr(idx << 4)
    dat = fix(dat, addr)
  return [addr, 0, dat, alt]


def create_brake_command(apply_brake, pcm_override, pcm_cancel_cmd, chime, idx):
  """Creates a CAN message for the Honda DBC BRAKE_COMMAND."""
  pump_on = apply_brake > 0
  brakelights = apply_brake > 0
  brake_rq = apply_brake > 0

  pcm_fault_cmd = False
  amount = struct.pack("!H", (apply_brake << 6) + pump_on)
  msg = amount + struct.pack("BBB", (pcm_override << 4) |
                             (pcm_fault_cmd << 2) |
                             (pcm_cancel_cmd << 1) | brake_rq, 0x80,
                             brakelights << 7) + chr(chime) + "\x00"
  return make_can_msg(0x1fa, msg, idx, 0)


def create_gas_command(gas_amount, idx):
  """Creates a CAN message for the Honda DBC GAS_COMMAND."""
  msg = struct.pack("!H", gas_amount)
  return make_can_msg(0x200, msg, idx, 0)


def create_steering_control(apply_steer, car_fingerprint, idx):
  """Creates a CAN message for the Honda DBC STEERING_CONTROL."""
  commands = []
  if car_fingerprint == CAR.CRV:
    msg_0x194 = struct.pack("!h", apply_steer << 4) + ("\x80" if apply_steer != 0 else "\x00")
    commands.append(make_can_msg(0x194, msg_0x194, idx, 0))
  else:
    msg_0xe4 = struct.pack("!h", apply_steer) + ("\x80\x00" if apply_steer != 0 else "\x00\x00")
    commands.append(make_can_msg(0xe4, msg_0xe4, idx, 0))
  return commands


def create_ui_commands(pcm_speed, hud, car_fingerprint, idx):
  """Creates an iterable of CAN messages for the UIs."""
  commands = []
  pcm_speed_real = np.clip(int(round(pcm_speed / 0.002759506)), 0,
                           64000)  # conversion factor from dbc file
  msg_0x30c = struct.pack("!HBBBBB", pcm_speed_real, hud.pcm_accel,
                          hud.v_cruise, hud.X2, hud.car, hud.X4)
  commands.append(make_can_msg(0x30c, msg_0x30c, idx, 0))

  msg_0x33d = chr(hud.X5) + chr(hud.lanes) + chr(hud.beep) + chr(hud.X8)
  commands.append(make_can_msg(0x33d, msg_0x33d, idx, 0))
  if car_fingerprint in (CAR.CIVIC, CAR.ODYSSEY):
    msg_0x35e = chr(0) * 7
    commands.append(make_can_msg(0x35e, msg_0x35e, idx, 0))
    msg_0x39f = (chr(0) * 2 + chr(hud.acc_alert) + chr(0) + chr(0xff) + chr(0x7f) + chr(0))
    commands.append(make_can_msg(0x39f, msg_0x39f, idx, 0))
  return commands


def create_radar_commands(v_ego, car_fingerprint, idx):
  """Creates an iterable of CAN messages for the radar system."""
  commands = []
  v_ego_kph = np.clip(int(round(v_ego * CV.MS_TO_KPH)), 0, 255)
  speed = struct.pack('!B', v_ego_kph)

  msg_0x300 = ("\xf9" + speed + "\x8a\xd0" +
               ("\x20" if idx == 0 or idx == 3 else "\x00") +
               "\x00\x00")

  if car_fingerprint == CAR.CIVIC:
    msg_0x301 = "\x02\x38\x44\x32\x4f\x00\x00"
    commands.append(make_can_msg(0x300, msg_0x300, idx + 8, 1))  # add 8 on idx.
  elif car_fingerprint == CAR.CRV:
    msg_0x301 = "\x00\x00\x50\x02\x51\x00\x00"
    commands.append(make_can_msg(0x300, msg_0x300, idx, 1))
  elif car_fingerprint == CAR.ODYSSEY:
    msg_0x301 = "\x00\x00\x56\x02\x55\x00\x00"
    commands.append(make_can_msg(0x300, msg_0x300, idx, 1))
  elif car_fingerprint == CAR.ACURA_ILX:
    msg_0x301 = "\x0f\x18\x51\x02\x5a\x00\x00"
    commands.append(make_can_msg(0x300, msg_0x300, idx, 1))

  commands.append(make_can_msg(0x301, msg_0x301, idx, 1))
  return commands
