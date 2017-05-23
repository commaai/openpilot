#!/usr/bin/env python
import os
import zmq
import numpy as np
import selfdrive.messaging as messaging

from selfdrive.services import service_list
from common.realtime import sec_since_boot, set_realtime_priority
from common.params import Params

from selfdrive.swaglog import cloudlog
from cereal import car

from selfdrive.controls.lib.pathplanner import PathPlanner
from selfdrive.controls.lib.adaptivecruise import AdaptiveCruise

def plannerd_thread(gctx):
  context = zmq.Context()
  poller = zmq.Poller()

  carstate = messaging.sub_sock(context, service_list['carState'].port, poller)
  live20 = messaging.sub_sock(context, service_list['live20'].port)
  model = messaging.sub_sock(context, service_list['model'].port)

  plan = messaging.pub_sock(context, service_list['plan'].port)

  # wait for stats about the car to come in from controls
  cloudlog.info("plannerd is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("plannerd got CarParams")

  CS = None
  PP = PathPlanner(model)
  AC = AdaptiveCruise(live20)

  # start the loop
  set_realtime_priority(2)

  # this runs whenever we get a packet that can change the plan
  while True:
    polld = poller.poll(timeout=1000)
    for sock, mode in polld:
      if mode != zmq.POLLIN or sock != carstate:
        continue

      cur_time = sec_since_boot()
      CS = messaging.recv_sock(carstate).carState

      PP.update(cur_time, CS.vEgo)

      # LoC.v_pid -> CS.vEgo
      # TODO: is this change okay?
      AC.update(cur_time, CS.vEgo, CS.steeringAngle, CS.vEgo, CP)

      # **** send the plan ****
      plan_send = messaging.new_message()
      plan_send.init('plan')

      # lateral plan
      plan_send.plan.lateralValid = not PP.dead
      if plan_send.plan.lateralValid:
        plan_send.plan.dPoly = map(float, PP.d_poly)

      # longitudal plan
      plan_send.plan.longitudinalValid = not AC.dead
      if plan_send.plan.longitudinalValid:
        plan_send.plan.vTarget = float(AC.v_target_lead)
        plan_send.plan.aTargetMin = float(AC.a_target[0])
        plan_send.plan.aTargetMax = float(AC.a_target[1])
        plan_send.plan.jerkFactor = float(AC.jerk_factor)
        plan_send.plan.hasLead = AC.has_lead

      plan.send(plan_send.to_bytes())


def main(gctx=None):
  plannerd_thread(gctx)

if __name__ == "__main__":
  main()

