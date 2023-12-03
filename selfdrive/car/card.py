#!/usr/bin/env python3
import os
import time

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.selfdrive.boardd.boardd import can_list_to_can_capnp
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can
from openpilot.system.swaglog import cloudlog

# a simple daemon for talking to cars

def main():
  sm = messaging.SubMaster(['carControl', 'pandaStates'])
  pm = messaging.PubMaster(['carState', 'sendcan'])

  can_timeout = None if os.environ.get('NO_CAN_TIMEOUT', False) else 20
  can = messaging.sub_sock('can', timeout=can_timeout)

  params = Params()
  config_realtime_process(4, Priority.CTRL_HIGH)

  # fingerprint
  cloudlog.warning("Waiting for CAN messages...")
  get_one_can(can)

  num_pandas = len(messaging.recv_one_retry(sm.sock['pandaStates']).pandaStates)
  CI, CP = get_car(can, pm.sock['sendcan'], params.get_bool("ExperimentalLongitudinalEnabled"), num_pandas)
  params.put("CarParams2", CP.to_bytes())

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
