#!/usr/bin/env python3
from cereal import car
from common.params import Params
from common.realtime import Priority, config_realtime_process
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.longitudinal_planner import Planner
from selfdrive.controls.lib.lateral_planner import LateralPlanner
import cereal.messaging as messaging


def plannerd_thread(sm=None, pm=None):

  config_realtime_process(2, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("plannerd got CarParams: %s", CP.carName)

  longitudinal_planner = Planner(CP)
  lateral_planner = LateralPlanner(CP)

  if sm is None:
    sm = messaging.SubMaster(['carState', 'controlsState', 'radarState', 'modelV2'],
                             poll=['radarState', 'modelV2'])

  if pm is None:
    pm = messaging.PubMaster(['longitudinalPlan', 'liveLongitudinalMpc', 'lateralPlan', 'liveMpc'])

  while True:
    sm.update()

    if sm.updated['modelV2']:
      lateral_planner.update(sm, CP)
      lateral_planner.publish(sm, pm)
    if sm.updated['radarState']:
      longitudinal_planner.update(sm, CP)
      longitudinal_planner.publish(sm, pm)


def main(sm=None, pm=None):
  plannerd_thread(sm, pm)


if __name__ == "__main__":
  main()
