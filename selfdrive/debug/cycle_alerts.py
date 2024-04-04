#!/usr/bin/env python3
import time
import random

from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.honda.interface import CarInterface
from openpilot.selfdrive.controls.lib.events import ET, Events
from openpilot.selfdrive.controls.lib.alertmanager import AlertManager
from openpilot.selfdrive.manager.process_config import managed_processes

EventName = car.CarEvent.EventName

def randperc() -> float:
  return 100. * random.random()

def cycle_alerts(duration=200, is_metric=False):
  # all alerts
  #alerts = list(EVENTS.keys())

  # this plays each type of audible alert
  alerts = [
    (EventName.buttonEnable, ET.ENABLE),
    (EventName.buttonCancel, ET.USER_DISABLE),
    (EventName.wrongGear, ET.NO_ENTRY),

    (EventName.locationdTemporaryError, ET.SOFT_DISABLE),
    (EventName.paramsdTemporaryError, ET.SOFT_DISABLE),
    (EventName.accFaulted, ET.IMMEDIATE_DISABLE),

    # DM sequence
    (EventName.preDriverDistracted, ET.WARNING),
    (EventName.promptDriverDistracted, ET.WARNING),
    (EventName.driverDistracted, ET.WARNING),
  ]

  # debug alerts
  alerts = [
    #(EventName.highCpuUsage, ET.NO_ENTRY),
    #(EventName.lowMemory, ET.PERMANENT),
    #(EventName.overheat, ET.PERMANENT),
    #(EventName.outOfSpace, ET.PERMANENT),
    #(EventName.modeldLagging, ET.PERMANENT),
    #(EventName.processNotRunning, ET.NO_ENTRY),
    #(EventName.commIssue, ET.NO_ENTRY),
    #(EventName.calibrationInvalid, ET.PERMANENT),
    (EventName.cameraMalfunction, ET.PERMANENT),
    (EventName.cameraFrameRate, ET.PERMANENT),
  ]

  cameras = ['roadCameraState', 'wideRoadCameraState', 'driverCameraState']

  CS = car.CarState.new_message()
  CP = CarInterface.get_non_essential_params("HONDA_CIVIC")
  sm = messaging.SubMaster(['deviceState', 'pandaStates', 'roadCameraState', 'modelV2', 'liveCalibration',
                            'driverMonitoringState', 'longitudinalPlan', 'liveLocationKalman',
                            'managerState'] + cameras)

  pm = messaging.PubMaster(['controlsState', 'pandaStates', 'deviceState'])

  events = Events()
  AM = AlertManager()

  frame = 0
  while True:
    for alert, et in alerts:
      events.clear()
      events.add(alert)

      sm['deviceState'].freeSpacePercent = randperc()
      sm['deviceState'].memoryUsagePercent = int(randperc())
      sm['deviceState'].cpuTempC = [randperc() for _ in range(3)]
      sm['deviceState'].gpuTempC = [randperc() for _ in range(3)]
      sm['deviceState'].cpuUsagePercent = [int(randperc()) for _ in range(8)]
      sm['modelV2'].frameDropPerc = randperc()

      if random.random() > 0.25:
        sm['modelV2'].velocity.x = [random.random(), ]
      if random.random() > 0.25:
        CS.vEgo = random.random()

      procs = [p.get_process_state_msg() for p in managed_processes.values()]
      random.shuffle(procs)
      for i in range(random.randint(0, 10)):
        procs[i].shouldBeRunning = True
      sm['managerState'].processes = procs

      sm['liveCalibration'].rpyCalib = [-1 * random.random() for _ in range(random.randint(0, 3))]

      for s in sm.data.keys():
        prob = 0.3 if s in cameras else 0.08
        sm.alive[s] = random.random() > prob
        sm.valid[s] = random.random() > prob
        sm.freq_ok[s] = random.random() > prob

      a = events.create_alerts([et, ], [CP, CS, sm, is_metric, 0])
      AM.add_many(frame, a)
      alert = AM.process_alerts(frame, [])
      print(alert)
      for _ in range(duration):
        dat = messaging.new_message()
        dat.init('controlsState')
        dat.controlsState.enabled = False

        if alert:
          dat.controlsState.alertText1 = alert.alert_text_1
          dat.controlsState.alertText2 = alert.alert_text_2
          dat.controlsState.alertSize = alert.alert_size
          dat.controlsState.alertStatus = alert.alert_status
          dat.controlsState.alertBlinkingRate = alert.alert_rate
          dat.controlsState.alertType = alert.alert_type
          dat.controlsState.alertSound = alert.audible_alert
        pm.send('controlsState', dat)

        dat = messaging.new_message()
        dat.init('deviceState')
        dat.deviceState.started = True
        pm.send('deviceState', dat)

        dat = messaging.new_message('pandaStates', 1)
        dat.pandaStates[0].ignitionLine = True
        dat.pandaStates[0].pandaType = log.PandaState.PandaType.uno
        pm.send('pandaStates', dat)

        frame += 1
        time.sleep(DT_CTRL)

if __name__ == '__main__':
  cycle_alerts()
