#!/usr/bin/env python
from common.numpy_fast import clip
from common.params import Params
from copy import copy
from cereal import car, log
import cereal.messaging as messaging
from selfdrive.car.car_helpers import get_car, get_one_can
from selfdrive.boardd.boardd import can_list_to_can_capnp

PandaType = log.HealthData.PandaType


def steer_thread():
  poller = messaging.Poller()

  logcan = messaging.sub_sock('can')
  joystick_sock = messaging.sub_sock('testJoystick', conflate=True, poller=poller)

  carstate = messaging.pub_sock('carState')
  carcontrol = messaging.pub_sock('carControl')
  sendcan = messaging.pub_sock('sendcan')

  button_1_last = 0
  enabled = False

  # wait for CAN packets
  print("Waiting for CAN messages...")
  get_one_can(logcan)

  CI, CP = get_car(logcan, sendcan)
  Params().put("CarParams", CP.to_bytes())

  CC = car.CarControl.new_message()

  while True:

    # send
    joystick = messaging.recv_one(joystick_sock)
    can_strs = messaging.drain_sock_raw(logcan, wait_for_one=True)
    CS = CI.update(CC, can_strs)

    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    actuators = car.CarControl.Actuators.new_message()

    if joystick is not None:
      axis_3 = clip(-joystick.testJoystick.axes[3] * 1.05, -1., 1.)          # -1 to 1
      actuators.steer = axis_3
      actuators.steerAngle = axis_3 * 43.   # deg
      axis_1 = clip(-joystick.testJoystick.axes[1] * 1.05, -1., 1.)          # -1 to 1
      actuators.gas = max(axis_1, 0.)
      actuators.brake = max(-axis_1, 0.)

      pcm_cancel_cmd = joystick.testJoystick.buttons[0]
      button_1 = joystick.testJoystick.buttons[1]
      if button_1 and not button_1_last:
        enabled = not enabled

      button_1_last = button_1

      #print "enable", enabled, "steer", actuators.steer, "accel", actuators.gas - actuators.brake

      hud_alert = 0
      if joystick.testJoystick.buttons[3]:
        hud_alert = "steerRequired"

    CC.actuators.gas = actuators.gas
    CC.actuators.brake = actuators.brake
    CC.actuators.steer = actuators.steer
    CC.actuators.steerAngle = actuators.steerAngle
    CC.hudControl.visualAlert = hud_alert
    CC.hudControl.setSpeed = 20
    CC.cruiseControl.cancel = pcm_cancel_cmd
    CC.enabled = enabled
    can_sends = CI.apply(CC)
    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan'))

    # broadcast carState
    cs_send = messaging.new_message('carState')
    cs_send.carState = copy(CS)
    carstate.send(cs_send.to_bytes())

    # broadcast carControl
    cc_send = messaging.new_message('carControl')
    cc_send.carControl = copy(CC)
    carcontrol.send(cc_send.to_bytes())


if __name__ == "__main__":
  steer_thread()
