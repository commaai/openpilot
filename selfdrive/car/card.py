#!/usr/bin/env python3
import time

import cereal.messaging as messaging
from panda import ALTERNATIVE_EXPERIENCE
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, DT_CTRL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.boardd.boardd import can_list_to_can_capnp
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can

# a simple daemon for talking to cars

def main():
  can = messaging.sub_sock('can', timeout=15)
  sm = messaging.SubMaster(['carControl', 'pandaStates'])
  pm = messaging.PubMaster(['carState', 'sendcan', 'carParams'])

  params = Params()
  config_realtime_process(4, Priority.CTRL_HIGH)

  # fingerprint
  cloudlog.warning("Waiting for CAN messages...")
  get_one_can(can)

  num_pandas = len(messaging.recv_one_retry(sm.sock['pandaStates']).pandaStates)
  CI, CP = get_car(can, pm.sock['sendcan'], params.get_bool("ExperimentalLongitudinalEnabled"), num_pandas)

  CP.alternativeExperience = 0
  if not params.get_bool("DisengageOnAccelerator"):
    CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS

  openpilot_enabled_toggle = params.get_bool("OpenpilotEnabledToggle")
  car_recognized = CP.carName != 'mock'
  controller_available = CI.CC is not None and openpilot_enabled_toggle and not CP.dashcamOnly
  CP.passive = not car_recognized or not controller_available or CP.dashcamOnly
  if CP.passive:
    CP.safetyConfigs = []

  # Write previous route's CarParams
  prev_cp = params.get("CarParamsPersistent")
  if prev_cp is not None:
    params.put("CarParamsPrevRoute", prev_cp)

  cp_bytes = CP.to_bytes()
  params.put("CarParams", cp_bytes)
  params.put("CarParamsCache", cp_bytes)
  params.put("CarParamsPersistent", cp_bytes)

  # cleanup params for options that aren't available anymore
  if not CP.experimentalLongitudinalAvailable:
    params.remove("ExperimentalLongitudinalEnabled")
  if not CP.openpilotLongitudinalControl:
    params.remove("ExperimentalMode")

  # driven by CAN
  initialized = False
  while True:
    can_strs = messaging.drain_sock_raw(can, wait_for_one=True)
    sm.update(0)

    # get a carState & publish it
    CS = CI.update(sm['carControl'], can_strs)
    cs_send = messaging.new_message(None, valid=CS.canValid, carState=CS)
    pm.send('carState', cs_send)

    # TODO: this can make us miss at least a few cycles on cars with experimental long
    # that need an ECU knockout
    if not initialized and params.get_bool('ControlsReady'):
      initialized = True
      CI.init(CP, can, pm.sock['sendcan'])

    # commands -> car bytes
    if not CP.passive and initialized:
      _, can_sends = CI.apply(sm['carControl'].as_builder(), int(time.monotonic() * 1e9))
      pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

    # carParams - logged every 50 seconds (> 1 per segment)
    if (sm.frame % int(50. / DT_CTRL) == 0):
      pm.send('carParams', messaging.new_message(None, valid=True, carParams=CP))


if __name__ == "__main__":
  main()
