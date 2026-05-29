#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from cereal.messaging import PubMaster, SubMaster
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.helpers import usbgpu_present


def main():
  sm = SubMaster(['bigModelV2', 'smolModelV2'])
  pm = PubMaster(['modelV2'])
  params = Params()
  usbgpu_present = usbgpu_present()
  src = 'bigModelV2' if usbgpu_present else 'smolModelV2'

  # fall back to smol once big has been seen and then lags; don't fall back if big was never seen
  big_stale_dt = 1.5 / SERVICE_LIST['bigModelV2'].frequency
  while True:
    sm.update()
    if src == 'bigModelV2':
      big_lagged = sm.seen['bigModelV2'] and (time.monotonic() - sm.recv_time['bigModelV2']) >= big_stale_dt
      if big_lagged:
        src = 'smolModelV2'
        params.put_bool("UsbGpuFailed", True)

    if sm.updated[src]:
      msg = messaging.new_message('modelV2')
      msg.valid = sm.valid[src]
      msg.modelV2 = sm[src]
      pm.send('modelV2', msg)


if __name__ == "__main__":
  main()
