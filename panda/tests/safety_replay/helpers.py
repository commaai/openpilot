import struct
import panda.tests.safety.libpandasafety_py as libpandasafety_py

safety_modes = {
  "NOOUTPUT": 0,
  "HONDA": 1,
  "TOYOTA": 2,
  "GM": 3,
  "HONDA_BOSCH": 4,
  "FORD": 5,
  "CADILLAC": 6,
  "HYUNDAI": 7,
  "TESLA": 8,
  "CHRYSLER": 9,
  "SUBARU": 10,
  "GM_ASCM": 0x1334,
  "TOYOTA_IPAS": 0x1335,
  "ALLOUTPUT": 0x1337,
  "ELM327": 0xE327
}

def to_signed(d, bits):
  ret = d
  if d >= (1 << (bits - 1)):
    ret = d - (1 << bits)
  return ret

def is_steering_msg(mode, addr):
  ret = False
  if mode == safety_modes["HONDA"] or mode == safety_modes["HONDA_BOSCH"]:
    ret = (addr == 0xE4) or (addr == 0x194) or (addr == 0x33D)
  elif mode == safety_modes["TOYOTA"]:
    ret = addr == 0x2E4
  elif mode == safety_modes["GM"]:
    ret = addr == 384
  elif mode == safety_modes["HYUNDAI"]:
    ret = addr == 832
  elif mode == safety_modes["CHRYSLER"]:
    ret = addr == 0x292
  elif mode == safety_modes["SUBARU"]:
    ret = addr == 0x122
  return ret

def get_steer_torque(mode, to_send):
  ret = 0
  if mode == safety_modes["HONDA"] or mode == safety_modes["HONDA_BOSCH"]:
    ret = to_send.RDLR & 0xFFFF0000
  elif mode == safety_modes["TOYOTA"]:
    ret = (to_send.RDLR & 0xFF00) | ((to_send.RDLR >> 16) & 0xFF)
    ret = to_signed(ret, 16)
  elif mode == safety_modes["GM"]:
    ret = ((to_send.RDLR & 0x7) << 8) + ((to_send.RDLR & 0xFF00) >> 8)
    ret = to_signed(ret, 11)
  elif mode == safety_modes["HYUNDAI"]:
    ret = ((to_send.RDLR >> 16) & 0x7ff) - 1024
  elif mode == safety_modes["CHRYSLER"]:
    ret = ((to_send.RDLR & 0x7) << 8) + ((to_send.RDLR & 0xFF00) >> 8) - 1024
  elif mode == safety_modes["SUBARU"]:
    ret = ((to_send.RDLR >> 16) & 0x1FFF)
    ret = to_signed(ret, 13)
  return ret

def set_desired_torque_last(safety, mode, torque):
  if mode == safety_modes["HONDA"] or mode == safety_modes["HONDA_BOSCH"]:
    pass # honda safety mode doesn't enforce a rate on steering msgs
  elif mode == safety_modes["TOYOTA"]:
    safety.set_toyota_desired_torque_last(torque)
  elif mode == safety_modes["GM"]:
    safety.set_gm_desired_torque_last(torque)
  elif mode == safety_modes["HYUNDAI"]:
    safety.set_hyundai_desired_torque_last(torque)
  elif mode == safety_modes["CHRYSLER"]:
    safety.set_chrysler_desired_torque_last(torque)
  elif mode == safety_modes["SUBARU"]:
    safety.set_subaru_desired_torque_last(torque)

def package_can_msg(msg):
  addr_shift = 3 if msg.address >= 0x800 else 21
  rdlr, rdhr = struct.unpack('II', msg.dat.ljust(8, b'\x00'))

  ret = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
  ret[0].RIR = msg.address << addr_shift
  ret[0].RDTR = len(msg.dat) | ((msg.src & 0xF) << 4)
  ret[0].RDHR = rdhr
  ret[0].RDLR = rdlr

  return ret

def init_segment(safety, lr, mode):
  sendcan = (msg for msg in lr if msg.which() == 'sendcan')
  steering_msgs = (can for msg in sendcan for can in msg.sendcan if is_steering_msg(mode, can.address))

  msg = next(steering_msgs, None)
  if msg is None:
    # no steering msgs
    return

  to_send = package_can_msg(msg)
  torque = get_steer_torque(mode, to_send)
  if torque != 0:
    safety.set_controls_allowed(1)
    set_desired_torque_last(safety, mode, torque)
    assert safety.safety_tx_hook(to_send), "failed to initialize panda safety for segment"

