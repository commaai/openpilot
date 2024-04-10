#!/usr/bin/env python3
import os
import time

import cereal.messaging as messaging

from cereal import car

from panda import ALTERNATIVE_EXPERIENCE

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL

from openpilot.selfdrive.boardd.boardd import can_list_to_can_capnp
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can
from openpilot.selfdrive.car.interfaces import CarInterfaceBase


REPLAY = "REPLAY" in os.environ


class CarD:
  CI: CarInterfaceBase
  CS: car.CarState

  def __init__(self, CI=None):
    self.can_sock = messaging.sub_sock('can', timeout=20)
    self.sm = messaging.SubMaster(['pandaStates'])
    self.pm = messaging.PubMaster(['sendcan', 'carState', 'carParams', 'carOutput'])

    self.can_rcv_timeout_counter = 0      # conseuctive timeout count
    self.can_rcv_cum_timeout_counter = 0  # cumulative timeout count

    self.CC_prev = car.CarControl.new_message()

    self.last_actuators = None

    self.params = Params()

    if CI is None:
      # wait for one pandaState and one CAN packet
      print("Waiting for CAN messages...")
      get_one_can(self.can_sock)

      num_pandas = len(messaging.recv_one_retry(self.sm.sock['pandaStates']).pandaStates)
      experimental_long_allowed = self.params.get_bool("ExperimentalLongitudinalEnabled")
      self.CI, self.CP = get_car(self.can_sock, self.pm.sock['sendcan'], experimental_long_allowed, num_pandas)
    else:
      self.CI, self.CP = CI, CI.CP

    # set alternative experiences from parameters
    disengage_on_accelerator = self.params.get_bool("DisengageOnAccelerator")
    self.CP.alternativeExperience = 0
    if not disengage_on_accelerator:
      self.CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS

    car_recognized = self.CP.carName != 'mock'
    openpilot_enabled_toggle = self.params.get_bool("OpenpilotEnabledToggle")

    controller_available = self.CI.CC is not None and openpilot_enabled_toggle and not self.CP.dashcamOnly

    self.CP.passive = not car_recognized or not controller_available or self.CP.dashcamOnly
    if self.CP.passive:
      safety_config = car.CarParams.SafetyConfig.new_message()
      safety_config.safetyModel = car.CarParams.SafetyModel.noOutput
      self.CP.safetyConfigs = [safety_config]

    # Write previous route's CarParams
    prev_cp = self.params.get("CarParamsPersistent")
    if prev_cp is not None:
      self.params.put("CarParamsPrevRoute", prev_cp)

    # Write CarParams for controls and radard
    cp_bytes = self.CP.to_bytes()
    self.params.put("CarParams", cp_bytes)
    self.params.put_nonblocking("CarParamsCache", cp_bytes)
    self.params.put_nonblocking("CarParamsPersistent", cp_bytes)

  def initialize(self):
    """Initialize CarInterface, once controls are ready"""
    self.CI.init(self.CP, self.can_sock, self.pm.sock['sendcan'])

  def state_update(self):
    """carState update loop, driven by can"""

    # Update carState from CAN
    can_strs = messaging.drain_sock_raw(self.can_sock, wait_for_one=True)
    self.CS = self.CI.update(self.CC_prev, can_strs)

    self.sm.update(0)

    can_rcv_valid = len(can_strs) > 0

    # Check for CAN timeout
    if not can_rcv_valid:
      self.can_rcv_timeout_counter += 1
      self.can_rcv_cum_timeout_counter += 1
    else:
      self.can_rcv_timeout_counter = 0

    self.can_rcv_timeout = self.can_rcv_timeout_counter >= 5

    if can_rcv_valid and REPLAY:
      self.can_log_mono_time = messaging.log_from_bytes(can_strs[0]).logMonoTime

    self.state_publish()

    return self.CS

  def state_publish(self):
    """carState and carParams publish loop"""

    # carState
    cs_send = messaging.new_message('carState')
    cs_send.valid = self.CS.canValid
    cs_send.carState = self.CS
    self.pm.send('carState', cs_send)

    # carParams - logged every 50 seconds (> 1 per segment)
    if (self.sm.frame % int(50. / DT_CTRL) == 0):
      cp_send = messaging.new_message('carParams')
      cp_send.valid = True
      cp_send.carParams = self.CP
      self.pm.send('carParams', cp_send)

    # publish new carOutput
    co_send = messaging.new_message('carOutput')
    co_send.valid = True
    if self.last_actuators is not None:
      co_send.carOutput.actuatorsOutput = self.last_actuators
    self.pm.send('carOutput', co_send)

  def controls_update(self, CC: car.CarControl):
    """control update loop, driven by carControl"""

    # send car controls over can
    now_nanos = self.can_log_mono_time if REPLAY else int(time.monotonic() * 1e9)
    self.last_actuators, can_sends = self.CI.apply(CC, now_nanos)
    self.pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=self.CS.canValid))

    self.CC_prev = CC

