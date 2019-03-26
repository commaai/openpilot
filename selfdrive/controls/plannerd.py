#!/usr/bin/env python
import zmq

from cereal import car
from common.params import Params
from selfdrive.swaglog import cloudlog
from selfdrive.services import service_list
from selfdrive.controls.lib.planner import Planner
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.pathplanner import PathPlanner
import selfdrive.messaging as messaging


def plannerd_thread():
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
  live100_sock = messaging.sub_sock(context, service_list['live100'].port, conflate=True, poller=poller)
  live20_sock = messaging.sub_sock(context, service_list['live20'].port, conflate=True, poller=poller)
  model_sock = messaging.sub_sock(context, service_list['model'].port, conflate=True, poller=poller)
  live_map_data_sock = messaging.sub_sock(context, service_list['liveMapData'].port, conflate=True, poller=poller)
  live_parameters_sock = messaging.sub_sock(context, service_list['liveParameters'].port, conflate=True, poller=poller)

  car_state = messaging.new_message()
  car_state.init('carState')
  live100 = messaging.new_message()
  live100.init('live100')
  model = messaging.new_message()
  model.init('model')
  live20 = messaging.new_message()
  live20.init('live20')
  live_map_data = messaging.new_message()
  live_map_data.init('liveMapData')

  live_parameters = messaging.new_message()
  live_parameters.init('liveParameters')
  live_parameters.liveParameters.valid = True
  live_parameters.liveParameters.steerRatio = CP.steerRatio
  live_parameters.liveParameters.stiffnessFactor = 1.0

  while True:
    for socket, event in poller.poll():
      if socket is live100_sock:
        live100 = messaging.recv_one(socket)
      elif socket is car_state_sock:
        car_state = messaging.recv_one(socket)
      elif socket is live_parameters_sock:
        live_parameters = messaging.recv_one(socket)
      elif socket is model_sock:
        model = messaging.recv_one(socket)
        PP.update(CP, VM, car_state, model, live100, live_parameters)
      elif socket is live_map_data_sock:
        live_map_data = messaging.recv_one(socket)
      elif socket is live20_sock:
        live20 = messaging.recv_one(socket)
        PL.update(car_state, CP, VM, PP, live20, live100, model, live_map_data)


def main(gctx=None):
  plannerd_thread()


if __name__ == "__main__":
  main()
