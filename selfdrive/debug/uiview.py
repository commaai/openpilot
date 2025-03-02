#!/usr/bin/env python3
import time

from cereal import car, log, messaging
from openpilot.common.params import Params
from openpilot.system.manager.process_config import managed_processes, is_snpe_model, is_tinygrad_model, is_stock_model
from openpilot.system.hardware import HARDWARE

if __name__ == "__main__":
  CP = car.CarParams(notCar=True, wheelbase=1, steerRatio=10)
  params = Params()
  params.put("CarParams", CP.to_bytes())

  if use_snpe_modeld := is_snpe_model(False, params, CP):
    print("Using SNPE modeld")
  if use_tinygrad_modeld := is_tinygrad_model(False, params, CP):
    print("Using TinyGrad modeld")
  if use_stock_modeld := is_stock_model(False, params, CP):
    print("Using stock modeld")

  HARDWARE.set_power_save(False)

  procs = ['camerad', 'ui', 'calibrationd', 'plannerd', 'dmonitoringmodeld', 'dmonitoringd']
  procs += ["modeld_snpe" if use_snpe_modeld else "modeld_tinygrad" if use_tinygrad_modeld else "modeld"]
  for p in procs:
    managed_processes[p].start()

  pm = messaging.PubMaster(['controlsState', 'deviceState', 'pandaStates', 'carParams'])

  msgs = {s: messaging.new_message(s) for s in ['controlsState', 'deviceState', 'carParams']}
  msgs['deviceState'].deviceState.started = True
  msgs['deviceState'].deviceState.deviceType = HARDWARE.get_device_type()
  msgs['carParams'].carParams.openpilotLongitudinalControl = True

  msgs['pandaStates'] = messaging.new_message('pandaStates', 1)
  msgs['pandaStates'].pandaStates[0].ignitionLine = True
  msgs['pandaStates'].pandaStates[0].pandaType = log.PandaState.PandaType.uno

  try:
    while True:
      time.sleep(1 / 100)  # continually send, rate doesn't matter
      for s in msgs:
        pm.send(s, msgs[s])
  except KeyboardInterrupt:
    for p in procs:
      managed_processes[p].stop()
