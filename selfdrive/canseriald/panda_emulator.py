from cereal import car, log, messaging


def emulate_panda(pm) -> None:
  pm.send("pandaStates", gen_panda_states())
  pm.send("peripheralState", gen_peripheral_state())


def gen_peripheral_state(voltage=12, current=0.3, fanSpeedRpm=1e3):
  msg = messaging.new_message('peripheralState')
  ps = msg.peripheralState
  ps.pandaType = log.PandaState.PandaType.dos
  ps.voltage = voltage * 1e3
  ps.current = current * 1e3
  ps.fanSpeedRpm = fanSpeedRpm
  ps.usbPowerMode = log.PeripheralState.UsbPowerMode.cdp
  return msg


def gen_panda_states(controlsAllowed=True):
  msg = messaging.new_message('pandaStates', 1)
  ps = msg.pandaStates[0]
  ps.ignitionLine = True
  ps.controlsAllowed = controlsAllowed
  ps.gasInterceptorDetected = False
  ps.canSendErrs = 0
  ps.canFwdErrs = 0
  ps.gmlanSendErrs = 0
  ps.pandaType = log.PandaState.PandaType.dos
  ps.ignitionCan = False
  ps.safetyModel = car.CarParams.SafetyModel.body
  ps.faultStatus = log.PandaState.FaultStatus.none
  ps.powerSaveEnabled = False
  ps.uptime = 1000
  ps.faults = []
  ps.canRxErrs = 0
  ps.harnessStatus = log.PandaState.HarnessStatus.flipped
  ps.heartbeatLost = False
  ps.alternativeExperience = 0
  ps.blockedCnt = 0
  ps.interruptLoad = 0
  ps.safetyParam = 0
  ps.alternativeExperience = 0
  ps.alternativeExperience = 0
  return msg
