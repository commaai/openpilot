#!/usr/bin/env python3
import time

import cereal.messaging as messaging
import selfdrive.manager as manager
from selfdrive.controls.lib.events import Alert


if __name__ == "__main__":

  pm = messaging.PubMaster(["controlsState", "thermal"])

  manager.prepare_managed_process("ui")
  manager.prepare_managed_process("camerad")
  manager.start_managed_process("ui")
  manager.start_managed_process("camerad")

  started_msg = messaging.new_message("thermalData")
  started_msg.started = True
  pm.send(started_msg, 'thermal')

  cnt = 0
  while True:
    msg = messaging.new_message('controlsState')
    msg.controlState.alertText1 = "DELAY TEST"
    msg.controlState.alertText2 = "#%d" % cnt
    msg.controlState.alertSize = 3
    msg.controlState.alertStatus = 1
    msg.controlState.alertSound = 7

    pm.send(msg, "controlState")
    pm.send(started_msg, 'thermal')
    cnt += 1
    time.sleep(0.01)
