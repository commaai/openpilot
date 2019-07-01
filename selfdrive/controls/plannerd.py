#!/usr/bin/env python
import gc

from cereal import car
from common.params import Params
from common.realtime import set_realtime_priority
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.planner import Planner
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.pathplanner import PathPlanner
import selfdrive.messaging as messaging


def plannerd_thread():
  gc.disable()

  # start the loop
  set_realtime_priority(2)

  params = Params()

  # Get FCW toggle from settings
  fcw_enabled = params.get("IsFcwEnabled") == "1"

  cloudlog.info("plannerd is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("plannerd got CarParams: %s", CP.carName)

  PL = Planner(CP, fcw_enabled)
  PP = PathPlanner(CP)

  VM = VehicleModel(CP)

  sm = messaging.SubMaster(['carState', 'controlsState', 'radarState', 'model', 'liveParameters'])

  sm['liveParameters'].valid = True
  sm['liveParameters'].sensorValid = True
  sm['liveParameters'].steerRatio = CP.steerRatio
  sm['liveParameters'].stiffnessFactor = 1.0
  live_map_data = messaging.new_message()
  live_map_data.init('liveMapData')

  while True:
    sm.update()

    if sm.updated['model']:
      PP.update(sm, CP, VM)
    if sm.updated['radarState']:
      PL.update(sm, CP, VM, PP, live_map_data.liveMapData)
    # elif socket is live_map_data_sock:
    #   live_map_data = msg


def main(gctx=None):
  plannerd_thread()


if __name__ == "__main__":
  main()
