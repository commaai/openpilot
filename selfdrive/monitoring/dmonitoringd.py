#!/usr/bin/env python3
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive.monitoring.helpers import DriverMonitoring


def dmonitoringd_thread():
  config_realtime_process([0, 1, 2, 3], 5)

  params = Params()
  pm = messaging.PubMaster(['driverMonitoringState'])
  sm = messaging.SubMaster(['driverStateV2', 'liveCalibration', 'carState', 'selfdriveState', 'modelV2',
                            'carControl'], poll='driverStateV2')

  DM = DriverMonitoring(rhd_saved=params.get_bool("IsRhdDetected"), always_on=params.get_bool("AlwaysOnDM"))

  # 20Hz <- dmonitoringmodeld
  while True:
    sm.update()
    if not sm.updated['driverStateV2']:
      # iterate when model has new output
      continue

    valid = sm.all_checks()
    if valid:
      DM.run_step(sm)

    # publish
    dat = DM.get_state_packet(valid=valid)
    pm.send('driverMonitoringState', dat)

    # load live always-on toggle
    if sm['driverStateV2'].frameId % 40 == 1:
      DM.always_on = params.get_bool("AlwaysOnDM")

    # save rhd virtual toggle every 5 mins
    if (sm['driverStateV2'].frameId % 6000 == 0 and
     DM.wheelpos_learner.filtered_stat.n > DM.settings._WHEELPOS_FILTER_MIN_COUNT and
     DM.wheel_on_right == (DM.wheelpos_learner.filtered_stat.M > DM.settings._WHEELPOS_THRESHOLD)):
      params.put_bool_nonblocking("IsRhdDetected", DM.wheel_on_right)

def main():
  dmonitoringd_thread()


if __name__ == '__main__':
  main()
