#!/usr/bin/env python3
import numpy as np
from cereal import car
from selfdrive.modeld.constants import T_IDXS
from common.params import Params
from common.realtime import Priority, config_realtime_process
from system.swaglog import cloudlog
from selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from selfdrive.controls.lib.lateral_planner import LateralPlanner
import cereal.messaging as messaging

def cumtrapz(x, t):
  return np.concatenate([[0], np.cumsum(((x[0:-1] + x[1:])/2) * np.diff(t))])

def plannerd_thread(sm=None, pm=None):
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = car.CarParams.from_bytes(params.get("CarParams", block=True))
  cloudlog.info("plannerd got CarParams: %s", CP.carName)

  longitudinal_planner = LongitudinalPlanner(CP)
  lateral_planner = LateralPlanner(CP)

  if sm is None:
    sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'radarState', 'modelV2'],
                             poll=['radarState', 'modelV2'], ignore_avg_freq=['radarState'])

  if pm is None:
    pm = messaging.PubMaster(['longitudinalPlan', 'lateralPlan', 'uiPlan'])

  while True:
    sm.update()

    if sm.updated['modelV2']:
      lateral_planner.update(sm)
      lateral_planner.publish(sm, pm)
      longitudinal_planner.update(sm)
      longitudinal_planner.publish(sm, pm)

      plan_odo = cumtrapz(longitudinal_planner.v_desired_trajectory_full, T_IDXS)
      model_odo = cumtrapz(lateral_planner.v_plan, T_IDXS)
      ui_x = np.interp(plan_odo, model_odo, lateral_planner.lat_mpc.x_sol[:,0])
      ui_y = np.interp(plan_odo, model_odo, lateral_planner.lat_mpc.x_sol[:,1])
      ui_z = np.interp(plan_odo, model_odo, lateral_planner.path_xyz[:,2])

      ui_send = messaging.new_message('uiPlan')
      ui_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])
      uiPlan = ui_send.uiPlan
      uiPlan.position.x = ui_x.tolist()
      uiPlan.position.y = ui_y.tolist()
      uiPlan.position.z = ui_z.tolist()
      pm.send('uiPlan', ui_send)


def main(sm=None, pm=None):
  plannerd_thread(sm, pm)


if __name__ == "__main__":
  main()
