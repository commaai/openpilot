#!/usr/bin/env python3
import time
import zmq
from hexdump import hexdump

import cereal.messaging as messaging
from cereal.services import service_list

if __name__ == "__main__":
  controls_state = messaging.pub_sock('controlsState')

  while 1:
    dat = messaging.new_message('controlsState')

    dat.controlsState.alertText1 = "alert text 1"
    dat.controlsState.alertText2 = "alert text 2"
    dat.controlsState.alertType = "test"
    dat.controlsState.alertSound = "chimeDisengage"
    controls_state.send(dat.to_bytes())

    time.sleep(0.01)
