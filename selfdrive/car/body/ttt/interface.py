#!/usr/bin/env python
import numpy as np
import cereal.messaging as messaging

from selfdrive.car.interfaces import CarInterfaceBase
from cereal import car

from selfdrive.boardd.boardd_api_impl import can_list_to_can_capnp  # pylint: disable=no-name-in-module,import-error

from opendbc.can.parser import CANParser
from opendbc.can.packer import CANPacker
packer = CANPacker("comma_body")

from selfdrive.car.interfaces import CarStateBase
class CarState(CarStateBase):
  def update(self, cp, v_ego):
    ret = car.CarState.new_message()
    ret.wheelSpeeds.fl = cp.vl['BODY_SENSOR']['SPEED_L']
    ret.wheelSpeeds.fr = cp.vl['BODY_SENSOR']['SPEED_R']
    ret.standstill = abs(v_ego) < 100
    return ret

  @staticmethod
  def get_can_parser(CP):
    return CANParser("comma_body", [("SPEED_L", "BODY_SENSOR", 0),
                                    ("SPEED_R", "BODY_SENSOR", 0)], [], enforce_checks=False)

class CarInterface(CarInterfaceBase):
  def update(self, c, can_strings, v_ego):
    self.cp.update_strings(can_strings)
    ret = self.CS.update(self.cp, v_ego)

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    print(f"Accel: {int(c.actuators.accel)}, steer: {int(c.actuators.steer)}")
    msg = packer.make_can_msg("BODY_COMMAND", 0,
      {"TORQUE_L": int(c.actuators.accel),
       "TORQUE_R": int(c.actuators.steer)})
    return [msg]

import atexit

if __name__ == "__main__":
  from common.realtime import Ratekeeper
  dtn = 80
  rk = Ratekeeper(dtn)

  can_sock = messaging.sub_sock('can')
  pm = messaging.PubMaster(['sendcan', 'carState'])
  CP = car.CarParams.new_message()
  ci = CarInterface(CP, None, CarState)

  def done():
    print("sending 0")
    ret = car.CarControl.new_message()
    msgs = ci.apply(ret)
    pm.send('sendcan', can_list_to_can_capnp(msgs, msgtype='sendcan'))

  atexit.register(done)

  sm = messaging.SubMaster(['sensorEvents', 'liveLocationKalman', 'testJoystick'])

  kp = 1300
  ki = 0
  kd = 280
  i = 0
  d = 0
  i_speed = 0
  i_tq = 0

  set_point = np.deg2rad(-0)

  accel_err = 0

  speed_measured = 0
  speed_desired = 0
  torque_right_filtered = 0.0
  torque_left_filtered = 0.0
  v_ego = 0.0
  while 1:
    sm.update()

    alpha = 1.0
    speed_desired = (1. - alpha)*speed_desired
    kp_speed = 0.001
    ki_speed = 0
    i_speed += ki_speed * (speed_desired - speed_measured)
    i_speed = np.clip(i_speed, -0.1, 0.1)
    set_point = kp_speed * (speed_desired - speed_measured) + i_speed
    try:
      angle_err = (-sm['liveLocationKalman'].orientationNED.value[1]) - set_point
      d_new = -sm['liveLocationKalman'].angularVelocityDevice.value[1]
      alpha_d = 1.0
      d = (1. - alpha_d) * d + alpha * d_new
      d =  np.clip(d, -1., 1.)
    except Exception:
      print("can't subscribe?")
      continue

    i += angle_err
    i = np.clip(i, -2, 2)

    can_strs = messaging.drain_sock_raw(can_sock, wait_for_one=False)
    cs = ci.update(None, can_strs, v_ego)
    cs_send = messaging.new_message('carState')
    cs_send.carState = cs
    pm.send('carState', cs_send)

    speed_measured = 0.5 * (cs.wheelSpeeds.fl + cs.wheelSpeeds.fr)
    v_ego_alpha = .03
    v_ego = (1.0 - v_ego_alpha) * v_ego + alpha * speed_measured

    ret = car.CarControl.new_message()
    speed = int(np.clip(angle_err*kp + accel_err*ki + d*kd, -1000, 1000))

    kp_diff = 0.95
    kd_diff = 0.1
    p_tq = (cs.wheelSpeeds.fl - cs.wheelSpeeds.fr)

    torque_diff = int(np.clip(p_tq*kp_diff + i_tq*kd_diff, -100, 100))

    i_tq += (cs.wheelSpeeds.fl - cs.wheelSpeeds.fr)
    print(f'diff : {torque_diff}')
    torque_r = speed + torque_diff
    torque_l = speed - torque_diff

    if torque_r > 0: torque_r += 10
    else: torque_r -= 10
    if torque_l > 0: torque_l += 10
    else: torque_l -= 10

    alpha_torque = 1.
    torque_right_filtered = (1. - alpha_torque) * torque_right_filtered + alpha_torque * torque_r
    torque_left_filtered = (1. - alpha_torque) * torque_left_filtered + alpha_torque * torque_l
    ret.actuators.steer = int(np.clip(torque_right_filtered, -1000, 1000))
    ret.actuators.accel = int(np.clip(torque_left_filtered, -1000, 1000))
    print(f'Set point : {set_point}')
    print(f'speeds : {cs.wheelSpeeds}')
    print(f'err angle : {angle_err}')
    accel_err += cs.wheelSpeeds.fl + cs.wheelSpeeds.fr
    print(f'accel_err : {accel_err}')
    print(f'd : {d}')
    msgs = ci.apply(ret)
    pm.send('sendcan', can_list_to_can_capnp(msgs, msgtype='sendcan'))
    rk.keep_time()
