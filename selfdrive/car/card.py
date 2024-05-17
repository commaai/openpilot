#!/usr/bin/env python3
import os
import time

import cereal.messaging as messaging

from cereal import car, log

from panda import ALTERNATIVE_EXPERIENCE

from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL

from openpilot.selfdrive.boardd.boardd import can_list_to_can_capnp
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can
from openpilot.selfdrive.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.controls.lib.events import Events

REPLAY = "REPLAY" in os.environ

State = log.ControlsState.OpenpilotState
EventName = car.CarEvent.EventName


def loop(CI=None):
  can_sock = messaging.sub_sock('can', timeout=20)
  sm = messaging.SubMaster(['pandaStates', 'carControl', 'controlsState'])
  pm = messaging.PubMaster(['sendcan', 'carState', 'carParams', 'carOutput'])

  can_rcv_timeout_counter = 0  # consecutive timeout count
  can_rcv_cum_timeout_counter = 0  # cumulative timeout count

  CC_prev = car.CarControl.new_message()
  CS_prev = car.CarState.new_message()
  controlsState_prev = car.CarState.new_message()

  last_actuators_output = car.CarControl.Actuators.new_message()

  params = Params()

  if CI is None:
    # wait for one pandaState and one CAN packet
    print("Waiting for CAN messages...")
    get_one_can(can_sock)

    num_pandas = len(messaging.recv_one_retry(sm.sock['pandaStates']).pandaStates)
    experimental_long_allowed = params.get_bool("ExperimentalLongitudinalEnabled")
    CI, CP = get_car(can_sock, pm.sock['sendcan'], experimental_long_allowed, num_pandas)
  else:
    CI, CP = CI, CI.CP

  # set alternative experiences from parameters
  disengage_on_accelerator = params.get_bool("DisengageOnAccelerator")
  CP.alternativeExperience = 0
  if not disengage_on_accelerator:
    CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS

  openpilot_enabled_toggle = params.get_bool("OpenpilotEnabledToggle")

  controller_available = CI.CC is not None and openpilot_enabled_toggle and not CP.dashcamOnly

  CP.passive = not controller_available or CP.dashcamOnly
  if CP.passive:
    safety_config = car.CarParams.SafetyConfig.new_message()
    safety_config.safetyModel = car.CarParams.SafetyModel.noOutput
    CP.safetyConfigs = [safety_config]

  # Write previous route's CarParams
  prev_cp = params.get("CarParamsPersistent")
  if prev_cp is not None:
    params.put("CarParamsPrevRoute", prev_cp)

  # Write CarParams for controls and radard
  cp_bytes = CP.to_bytes()
  params.put("CarParams", cp_bytes)
  params.put_nonblocking("CarParamsCache", cp_bytes)
  params.put_nonblocking("CarParamsPersistent", cp_bytes)

  events = Events()

  # card is driven by can recv, expected at 100Hz
  rk = Ratekeeper(100, print_delay_threshold=None)

  while True:
    # *** state update ***
    """carState update loop, driven by can"""

    # Update carState from CAN
    can_strs = messaging.drain_sock_raw(can_sock, wait_for_one=True)
    CS = CI.update(CC_prev, can_strs)

    sm.update(0)

    can_rcv_valid = len(can_strs) > 0

    # Check for CAN timeout
    if not can_rcv_valid:
      can_rcv_timeout_counter += 1
      can_rcv_cum_timeout_counter += 1
    else:
      can_rcv_timeout_counter = 0

    can_rcv_timeout = can_rcv_timeout_counter >= 5

    if can_rcv_valid and REPLAY:
      can_log_mono_time = messaging.log_from_bytes(can_strs[0]).logMonoTime

    # *** update events ***
    events.clear()

    events.add_from_msg(CS.events)

    # Disable on rising edge of accelerator or brake. Also disable on brake when speed > 0
    if (CS.gasPressed and not CS_prev.gasPressed and disengage_on_accelerator) or \
      (CS.brakePressed and (not CS_prev.brakePressed or not CS.standstill)) or \
      (CS.regenBraking and (not CS_prev.regenBraking or not CS.standstill)):
      events.add(EventName.pedalPressed)

    CS.events = events.to_msg()

    # *** state publish ***
    """carState and carParams publish loop"""

    # carParams - logged every 50 seconds (> 1 per segment)
    if sm.frame % int(50. / DT_CTRL) == 0:
      cp_send = messaging.new_message('carParams')
      cp_send.valid = True
      cp_send.carParams = CP
      pm.send('carParams', cp_send)

    # publish new carOutput
    co_send = messaging.new_message('carOutput')
    co_send.valid = sm.all_checks(['carControl'])
    co_send.carOutput.actuatorsOutput = last_actuators_output
    pm.send('carOutput', co_send)

    # kick off controlsd step while we actuate the latest carControl packet
    cs_send = messaging.new_message('carState')
    cs_send.valid = CS.canValid
    cs_send.carState = CS
    cs_send.carState.canRcvTimeout = can_rcv_timeout
    cs_send.carState.canErrorCounter = can_rcv_cum_timeout_counter
    cs_send.carState.cumLagMs = -rk.remaining * 1000.
    pm.send('carState', cs_send)

    # *** controls update ***
    controlsState = sm['controlsState']
    if not CP.passive and controlsState.initialized:
      """control update loop, driven by carControl"""

      if not controlsState_prev.initialized:
        # Initialize CarInterface, once controls are ready
        CI.init(CP, can_sock, pm.sock['sendcan'])

      if sm.all_checks(['carControl']):
        # send car controls over can
        CC = sm['carControl']
        now_nanos = can_log_mono_time if REPLAY else int(time.monotonic() * 1e9)
        last_actuators_output, can_sends = CI.apply(CC, now_nanos)
        pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

        CC_prev = CC

    controlsState_prev = controlsState
    CS_prev = CS.as_reader()

    rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  loop()


if __name__ == "__main__":
  main()
