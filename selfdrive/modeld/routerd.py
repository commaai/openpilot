#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from cereal.messaging import PubMaster, SubMaster
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive.modeld.fill_model_msg import fill_driving_model_data

MAX_WARMUP_TIME = 0.5
MAX_DT = 0.05

def main():
  config_realtime_process(7, 54)

  sm = SubMaster(['bigModelV2', 'smolModelV2', 'usbgpuState'])
  pm = PubMaster(['modelV2', 'modelSource', 'drivingModelData'])
  params = Params()

  while not sm.seen['usbgpuState']:
    sm.update()
  src = 'bigModelV2' if (sm['usbgpuState'].usbgpuPresent and params.get_bool("UsbgpuCompiled")) else 'smolModelV2'

  while True:
    sm.update()
    if sm.updated['bigModelV2'] and big_first_t is None:
      big_first_t = time.monotonic()

    if src == 'bigModelV2':
      settled = big_first_t is not None and (time.monotonic() - big_first_t) >= MAX_WARMUP_TIME
      big_lagged = settled and (time.monotonic() - sm.recv_time['bigModelV2']) >= MAX_DT
      if big_lagged or not sm['usbgpuState'].usbgpuPresent:
        src = 'smolModelV2'
        params.put_bool("UsbgpuFailed", True)

    if sm.updated[src]:
      msg = messaging.new_message('modelV2')
      msg.valid = sm.valid[src]
      msg.modelV2 = sm[src]

      drivingdata_send = messaging.new_message('drivingModelData')
      fill_driving_model_data(drivingdata_send, msg)

      source = messaging.new_message('modelSource', valid=True)
      source.modelSource.big = src == 'bigModelV2'

      pm.send('modelV2', msg)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('modelSource', source)


if __name__ == "__main__":
  main()
