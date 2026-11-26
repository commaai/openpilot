#!/usr/bin/env python3
import copy
import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive.monitoring.policy import DriverMonitoring


def new_demo_dm(DM: DriverMonitoring) -> DriverMonitoring:
  # the driver camera preview starts fresh each time, keeping the learned wheel side
  demo_DM = DriverMonitoring(rhd_saved=DM.wheel_on_right, demo=True)
  demo_DM.wheelpos_offsetter = copy.deepcopy(DM.wheelpos_offsetter)
  return demo_DM


def dmonitoringd_thread():
  config_realtime_process([0, 1, 2, 3], 5)

  params = Params()
  pm = messaging.PubMaster(['driverMonitoringState'])
  sm = messaging.SubMaster(['driverStateV2', 'extrinsicsCalibration', 'carState', 'selfdriveState', 'modelV2'], poll='driverStateV2')

  DM = DriverMonitoring(rhd_saved=params.get_bool("IsRhdDetected"), always_on=params.get_bool("AlwaysOnDM"))
  demo_mode=False
  demo_DM = None

  # 20Hz <- dmonitoringmodeld
  while True:
    sm.update()
    if not sm.updated['driverStateV2']:
      # iterate when model has new output
      continue

    valid = sm.all_checks()
    if demo_DM is not None and sm.valid['driverStateV2']:
      if params.get_bool("DriverViewReset"):
        params.remove("DriverViewReset")
        demo_DM = new_demo_dm(DM)
      demo_DM.run_step(sm, demo=True)
    elif valid:
      DM.run_step(sm)

    # publish
    dat = (demo_DM or DM).get_state_packet(valid=valid)
    pm.send('driverMonitoringState', dat)

    # load live always-on toggle
    if sm['driverStateV2'].frameId % 40 == 1:
      DM.always_on = params.get_bool("AlwaysOnDM")
      demo_mode = params.get_bool("IsDriverViewEnabled")
      if demo_mode != (demo_DM is not None):
        demo_DM = new_demo_dm(DM) if demo_mode else None

    # save rhd virtual toggle every 5 mins
    if (sm['driverStateV2'].frameId % 6000 == 0 and not demo_mode and
     DM.wheelpos_offsetter.filtered_stat.n > DM.settings._WHEELPOS_FILTER_MIN_COUNT and
     DM.wheel_on_right == (DM.wheelpos_offsetter.filtered_stat.M > DM.settings._WHEELPOS_THRESHOLD)):
      params.put_bool("IsRhdDetected", DM.wheel_on_right)

def main():
  dmonitoringd_thread()


if __name__ == '__main__':
  main()
