#!/usr/bin/env python3
import time

from cereal import car
import cereal.messaging as messaging
from panda import ALTERNATIVE_EXPERIENCE
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.selfdrive.boardd.boardd import can_list_to_can_capnp
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can
from openpilot.system.swaglog import cloudlog

# a simple daemon for talking to cars

def main():
  can = messaging.sub_sock('can', timeout=20)
  sm = messaging.SubMaster(['carControl', 'pandaStates'])
  pm = messaging.PubMaster(['carState', 'sendcan'])

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

  car_recognized = CP.carName != 'mock'
  passive = params.get_bool("Passive") or not params.get_bool("OpenpilotEnabledToggle")
  controller_available = CI.CC is not None and not passive and not CP.dashcamOnly
  read_only = not car_recognized or not controller_available or CP.dashcamOnly
  if read_only:
    safety_config = car.CarParams.SafetyConfig.new_message()
    safety_config.safetyModel = car.CarParams.SafetyModel.noOutput
    CP.safetyConfigs = [safety_config]

  # Write previous route's CarParams
  prev_cp = params.get("CarParamsPersistent")
  if prev_cp is not None:
    params.put("CarParamsPrevRoute", prev_cp)

  cp_bytes = CP.to_bytes()
  params.put("CarParams", cp_bytes)
  params.put("CarParamsCache", cp_bytes)
  params.put("CarParamsPersistent", cp_bytes)

  rk = Ratekeeper(int(1 / DT_CTRL), print_delay_threshold=None)
  while True:
    can_strs = messaging.drain_sock_raw(can, wait_for_one=True)
    sm.update(0)

    # get a carState & publish it
    CS = CI.update(sm['carControl'], can_strs)
    cs_send = messaging.new_message(None, valid=CS.canValid, carState=CS)
    pm.send('carState', cs_send)

    # commands -> car bytes
    _, can_sends = CI.apply(sm['carControl'].as_builder(), int(time.monotonic() * 1e9))
    pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

    rk.keep_time()

if __name__ == "__main__":
  main()
