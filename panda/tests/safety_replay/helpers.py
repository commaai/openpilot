#!/usr/bin/env python3
import struct
import panda.tests.safety.libpandasafety_py as libpandasafety_py
from panda import Panda

def to_signed(d, bits):
  ret = d
  if d >= (1 << (bits - 1)):
    ret = d - (1 << bits)
  return ret

def is_steering_msg(mode, addr):
  ret = False
  if mode in [Panda.SAFETY_HONDA_NIDEC, Panda.SAFETY_HONDA_BOSCH_GIRAFFE, Panda.SAFETY_HONDA_BOSCH_HARNESS]:
    ret = (addr == 0xE4) or (addr == 0x194) or (addr == 0x33D)
  elif mode == Panda.SAFETY_TOYOTA:
    ret = addr == 0x2E4
  elif mode == Panda.SAFETY_GM:
    ret = addr == 384
  elif mode == Panda.SAFETY_HYUNDAI:
    ret = addr == 832
  elif mode == Panda.SAFETY_CHRYSLER:
    ret = addr == 0x292
  elif mode == Panda.SAFETY_SUBARU:
    ret = addr == 0x122
  return ret

def get_steer_torque(mode, to_send):
  ret = 0
  if mode in [Panda.SAFETY_HONDA_NIDEC, Panda.SAFETY_HONDA_BOSCH_GIRAFFE, Panda.SAFETY_HONDA_BOSCH_HARNESS]:
    ret = to_send.RDLR & 0xFFFF0000
  elif mode == Panda.SAFETY_TOYOTA:
    ret = (to_send.RDLR & 0xFF00) | ((to_send.RDLR >> 16) & 0xFF)
    ret = to_signed(ret, 16)
  elif mode == Panda.SAFETY_GM:
    ret = ((to_send.RDLR & 0x7) << 8) + ((to_send.RDLR & 0xFF00) >> 8)
    ret = to_signed(ret, 11)
  elif mode == Panda.SAFETY_HYUNDAI:
    ret = ((to_send.RDLR >> 16) & 0x7ff) - 1024
  elif mode == Panda.SAFETY_CHRYSLER:
    ret = ((to_send.RDLR & 0x7) << 8) + ((to_send.RDLR & 0xFF00) >> 8) - 1024
  elif mode == Panda.SAFETY_SUBARU:
    ret = ((to_send.RDLR >> 16) & 0x1FFF)
    ret = to_signed(ret, 13)
  return ret

def set_desired_torque_last(safety, mode, torque):
  if mode in [Panda.SAFETY_HONDA_NIDEC, Panda.SAFETY_HONDA_BOSCH_GIRAFFE, Panda.SAFETY_HONDA_BOSCH_HARNESS]:
    pass # honda safety mode doesn't enforce a rate on steering msgs
  elif mode == Panda.SAFETY_TOYOTA:
    safety.set_toyota_desired_torque_last(torque)
  elif mode == Panda.SAFETY_GM:
    safety.set_gm_desired_torque_last(torque)
  elif mode == Panda.SAFETY_HYUNDAI:
    safety.set_hyundai_desired_torque_last(torque)
  elif mode == Panda.SAFETY_CHRYSLER:
    safety.set_chrysler_desired_torque_last(torque)
  elif mode == Panda.SAFETY_SUBARU:
    safety.set_subaru_desired_torque_last(torque)

def package_can_msg(msg):
  rdlr, rdhr = struct.unpack('II', msg.dat.ljust(8, b'\x00'))

  ret = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
  if msg.address >= 0x800:
    ret[0].RIR = (msg.address << 3) | 5
  else:
    ret[0].RIR = (msg.address << 21) | 1
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

