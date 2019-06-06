#!/usr/bin/env python
import gc
import zmq
from collections import defaultdict

from cereal import car
from common.params import Params
from common.realtime import sec_since_boot, set_realtime_priority
from selfdrive.swaglog import cloudlog
from selfdrive.services import service_list
from selfdrive.controls.lib.planner import Planner
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.pathplanner import PathPlanner
import selfdrive.messaging as messaging


def plannerd_thread():
  gc.disable()

  # start the loop
  set_realtime_priority(2)

  context = zmq.Context()
  params = Params()

  # Get FCW toggle from settings
  fcw_enabled = params.get("IsFcwEnabled") == "1"

  cloudlog.info("plannerd is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("plannerd got CarParams: %s", CP.carName)

  PL = Planner(CP, fcw_enabled)
  PP = PathPlanner(CP)

  VM = VehicleModel(CP)

  poller = zmq.Poller()
  car_state_sock = messaging.sub_sock(context, service_list['carState'].port, conflate=True, poller=poller)
  controls_state_sock = messaging.sub_sock(context, service_list['controlsState'].port, conflate=True, poller=poller)
  radar_state_sock = messaging.sub_sock(context, service_list['radarState'].port, conflate=True, poller=poller)
  model_sock = messaging.sub_sock(context, service_list['model'].port, conflate=True, poller=poller)
  live_parameters_sock = messaging.sub_sock(context, service_list['liveParameters'].port, conflate=True, poller=poller)
  # live_map_data_sock = messaging.sub_sock(context, service_list['liveMapData'].port, conflate=True, poller=poller)

  car_state = messaging.new_message()
  car_state.init('carState')
  controls_state = messaging.new_message()
  controls_state.init('controlsState')
  model = messaging.new_message()
  model.init('model')
  radar_state = messaging.new_message()
  radar_state.init('radarState')
  live_map_data = messaging.new_message()
  live_map_data.init('liveMapData')

  live_parameters = messaging.new_message()
  live_parameters.init('liveParameters')
  live_parameters.liveParameters.valid = True
  live_parameters.liveParameters.sensorValid = True
  live_parameters.liveParameters.steerRatio = CP.steerRatio
  live_parameters.liveParameters.stiffnessFactor = 1.0

  rcv_times = defaultdict(int)

  while True:
    for socket, event in poller.poll():
      msg = messaging.recv_one(socket)
      rcv_times[msg.which()] = sec_since_boot()

      if socket is controls_state_sock:
        controls_state = msg
      elif socket is car_state_sock:
        car_state = msg
      elif socket is live_parameters_sock:
        live_parameters = msg
      elif socket is model_sock:
        model = msg
        PP.update(rcv_times, CP, VM, car_state, model, controls_state, live_parameters)
      elif socket is radar_state_sock:
        radar_state = msg
        PL.update(rcv_times, car_state, CP, VM, PP, radar_state, controls_state, model, live_map_data)
      # elif socket is live_map_data_sock:
      #   live_map_data = msg


def main(gctx=None):
  plannerd_thread()


if __name__ == "__main__":
  main()
