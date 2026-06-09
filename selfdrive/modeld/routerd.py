#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from cereal.messaging import PubMaster, SubMaster
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process


def main():
  config_realtime_process(7, 54)

  sm = SubMaster(['bigModelV2', 'smolModelV2', 'usbgpuState'])
  pm = PubMaster(['modelV2', 'modelSource'])
  params = Params()

  # decide the initial source from the first usbgpuState
  while not sm.seen['usbgpuState']:
    sm.update()
  src = 'bigModelV2' if (sm['usbgpuState'].usbgpuPresent and sm['usbgpuState'].usbgpuCompiled) else 'smolModelV2'

  # fall back to smol if the usbgpu disappears, or once big has been seen and then lags
  big_stale_dt = 1.5 / SERVICE_LIST['bigModelV2'].frequency
  while True:
    sm.update()
    if src == 'bigModelV2':
      big_lagged = sm.seen['bigModelV2'] and (time.monotonic() - sm.recv_time['bigModelV2']) >= big_stale_dt
      if big_lagged or not sm['usbgpuState'].usbgpuPresent:
        src = 'smolModelV2'
        params.put_bool("UsbgpuFailed", True)

    if sm.updated[src]:
      msg = messaging.new_message('modelV2')
      msg.valid = sm.valid[src]
      msg.modelV2 = sm[src]
      pm.send('modelV2', msg)

      source = messaging.new_message('modelSource', valid=True)
      source.modelSource.big = src == 'bigModelV2'
      pm.send('modelSource', source)

  # TODO prune modelv2 to make drivingModelData
  # and forward odom


if __name__ == "__main__":
  main()
