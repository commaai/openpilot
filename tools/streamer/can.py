#!/usr/bin/env python3
from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from selfdrive.boardd.boardd_api_impl import can_list_to_can_capnp  # pylint: disable=no-name-in-module,import-error



packer = CANPacker("simulator_dbc")

def get_sim_can_parser():
  signals = [
    # sig_name, sig_address
    ("STEER_ANGLE", "STEER"),
    ("CRZ_ACTIVE", "CRZ_CTRL"),
    ("CRZ_SPEED", "CRZ_EVENTS"),
    ("STANDSTILL", "PEDALS"),
  ]

  checks = [
    # sig_address, frequency
    ("STEER", 67),
    ("ENGINE_DATA", 100),
    ("CRZ_CTRL", 50),
    ("CRZ_EVENTS", 50),
    ("PEDALS", 50),

  ]

  return CANParser("simulator_dbc", signals, checks, 0)

cp = get_sim_can_parser()

def simulator_can_function(pm, speed, angle, idx, cruise_sp, is_engaged):
  msg = []
  #
  # cp.vl["ENGINE_DATA"]["SPEED"]
  msg.append(packer.make_can_msg("ENGINE_DATA", 0, {"SPEED": speed}))
  # cp.vl["STEER"]["STEER_ANGLE"]
  msg.append(packer.make_can_msg("STEER", 0, {"STEER_ANGLE": angle}))
  # cp.vl["CRZ_CTRL"]["CRZ_ACTIVE"] == 1
  msg.append(packer.make_can_msg("CRZ_CTRL", 0, {"CRZ_ACTIVE": is_engaged}))
  # cp.vl["PEDALS"]["STANDSTILL"] == 1
  msg.append(packer.make_can_msg("PEDALS", 0, {"STANDSTILL": 1 if speed < 0.1 else 0}))
  # ret.cruiseState.speed = cp.vl["CRZ_EVENTS"]["CRZ_SPEED"] * CV.KPH_TO_MS
  msg.append(packer.make_can_msg("CRZ_EVENTS", 0, {"CRZ_SPEED": cruise_sp}))

  pm.send('can', can_list_to_can_capnp(msg))
