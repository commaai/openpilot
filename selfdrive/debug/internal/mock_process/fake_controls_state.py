#!/usr/bin/env python3
import time
import zmq
from hexdump import hexdump

from common.realtime import Ratekeeper
import cereal.messaging as messaging
from cereal.services import service_list

if __name__ == "__main__":
  controls_state = messaging.pub_sock('controlsState')

  rk = Ratekeeper(100)
  while 1:
    dat = messaging.new_message()
    dat.init('controlsState')

    dat.controlsState.vEgo = 25.
    dat.controlsState.enabled = True
    controls_state.send(dat.to_bytes())

    rk.keep_time()
