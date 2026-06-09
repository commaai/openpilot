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

  # fall back to smol if the usbgpu disappears, or once big has settled and then lags
  big_stale_dt = 1.5 / SERVICE_LIST['bigModelV2'].frequency
  big_grace_dt = 5.  # ignore big lagging for a bit after it first comes up
  big_first_t = None
  while True:
    sm.update()
    if sm.updated['bigModelV2'] and big_first_t is None:
      big_first_t = time.monotonic()

    if src == 'bigModelV2':
      settled = big_first_t is not None and (time.monotonic() - big_first_t) >= big_grace_dt
      big_lagged = settled and (time.monotonic() - sm.recv_time['bigModelV2']) >= big_stale_dt
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
