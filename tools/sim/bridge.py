#!/usr/bin/env python3
import carla
import os
import time, termios, tty, sys
import math
import atexit
import numpy as np
import threading
import random
import cereal.messaging as messaging
import argparse
import queue
from common.params import Params
from common.realtime import Ratekeeper
from lib.can import can_function, sendcan_function
from lib.helpers import FakeSteeringWheel
from selfdrive.car.honda.values import CruiseButtons

parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
parser.add_argument('--autopilot', action='store_true')
parser.add_argument('--joystick', action='store_true')
parser.add_argument('--realmonitoring', action='store_true')
args = parser.parse_args()

pm = messaging.PubMaster(['frame', 'sensorEvents', 'can'])

W,H = 1164, 874

def cam_callback(image):
  img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
  img = np.reshape(img, (H, W, 4))
  img = img[:, :, [0,1,2]].copy()

  dat = messaging.new_message('frame')
  dat.frame = {
    "frameId": image.frame,
    "image": img.tostring(),
    "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  }
  pm.send('frame', dat)

def imu_callback(imu):
  #print(imu, imu.accelerometer)

  dat = messaging.new_message('sensorEvents', 2)
  dat.sensorEvents[0].sensor = 4
  dat.sensorEvents[0].type = 0x10
  dat.sensorEvents[0].init('acceleration')
  dat.sensorEvents[0].acceleration.v = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
  # copied these numbers from locationd
  dat.sensorEvents[1].sensor = 5
  dat.sensorEvents[1].type = 0x10
  dat.sensorEvents[1].init('gyroUncalibrated')
  dat.sensorEvents[1].gyroUncalibrated.v = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
  pm.send('sensorEvents', dat)

def health_function():
  pm = messaging.PubMaster(['health'])
  rk = Ratekeeper(1.0)
  while 1:
    dat = messaging.new_message('health')
    dat.valid = True
    dat.health = {
      'ignitionLine': True,
      'hwType': "whitePanda",
      'controlsAllowed': True
    }
    pm.send('health', dat)
    rk.keep_time()

def fake_driver_monitoring():
  if args.realmonitoring:
    return
  pm = messaging.PubMaster(['driverState'])
  while 1:
    dat = messaging.new_message('driverState')
    dat.driverState.faceProb = 1.0
    pm.send('driverState', dat)
    time.sleep(0.1)

def go(q):
  threading.Thread(target=health_function).start()
  threading.Thread(target=fake_driver_monitoring).start()

  import carla
  client = carla.Client("127.0.0.1", 2000)
  client.set_timeout(5.0)
  world = client.load_world('Town04')
  settings = world.get_settings()
  settings.fixed_delta_seconds = 0.05
  world.apply_settings(settings)

  weather = carla.WeatherParameters(
      cloudyness=0.1,
      precipitation=0.0,
      precipitation_deposits=0.0,
      wind_intensity=0.0,
      sun_azimuth_angle=15.0,
      sun_altitude_angle=75.0)
  world.set_weather(weather)

  blueprint_library = world.get_blueprint_library()
  """
  for blueprint in blueprint_library.filter('sensor.*'):
     print(blueprint.id)
  exit(0)
  """

  world_map = world.get_map()
  vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.*'))
  vehicle = world.spawn_actor(vehicle_bp, world_map.get_spawn_points()[16])

  # make tires less slippery
  wheel_control = carla.WheelPhysicsControl(tire_friction=5)
  physics_control = vehicle.get_physics_control()
  physics_control.mass = 1326
  physics_control.wheels = [wheel_control]*4
  physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
  physics_control.gear_switch_time = 0.0
  vehicle.apply_physics_control(physics_control)

  if args.autopilot:
    vehicle.set_autopilot(True)
  # print(vehicle.get_speed_limit())

  blueprint = blueprint_library.find('sensor.camera.rgb')
  blueprint.set_attribute('image_size_x', str(W))
  blueprint.set_attribute('image_size_y', str(H))
  blueprint.set_attribute('fov', '70')
  blueprint.set_attribute('sensor_tick', '0.05')
  transform = carla.Transform(carla.Location(x=0.8, z=1.45))
  camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
  camera.listen(cam_callback)

  # reenable IMU
  imu_bp = blueprint_library.find('sensor.other.imu')
  imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
  imu.listen(imu_callback)

  def destroy():
    print("clean exit")
    imu.destroy()
    camera.destroy()
    vehicle.destroy()
    print("done")
  atexit.register(destroy)

  # can loop
  sendcan = messaging.sub_sock('sendcan')
  rk = Ratekeeper(100, print_delay_threshold=0.05)

  # init
  A_throttle = 2.
  A_brake = 2.
  A_steer_torque = 1.
  fake_wheel = FakeSteeringWheel()
  is_openpilot_engaged = False
  in_reverse = False

  throttle_out = 0
  brake_out = 0
  steer_angle_out = 0

  while 1:
    cruise_button = 0

    # check for a input message, this will not block
    if not q.empty():
      print("here")
      message = q.get()

      m = message.split('_')
      if m[0] == "steer":
        steer_angle_out = float(m[1])
        fake_wheel.set_angle(steer_angle_out) # touching the wheel overrides fake wheel angle
        # print(" === steering overriden === ")
      if m[0] == "throttle":
        throttle_out = float(m[1]) / 100.
        if throttle_out > 0.3:
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
      if m[0] == "brake":
        brake_out = float(m[1]) / 100.
        if brake_out > 0.3:
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
      if m[0] == "reverse":
        in_reverse = not in_reverse
        cruise_button = CruiseButtons.CANCEL
        is_openpilot_engaged = False
      if m[0] == "cruise":
        if m[1] == "down":
          cruise_button = CruiseButtons.DECEL_SET
          is_openpilot_engaged = True
        if m[1] == "up":
          cruise_button = CruiseButtons.RES_ACCEL
          is_openpilot_engaged = True
        if m[1] == "cancel":
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False

    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
    can_function(pm, speed, fake_wheel.angle, rk.frame, cruise_button=cruise_button, is_engaged=is_openpilot_engaged)

    if rk.frame%1 == 0: # 20Hz?
      throttle_op, brake_op, steer_torque_op = sendcan_function(sendcan)
      # print(" === torq, ",steer_torque_op, " ===")
      if is_openpilot_engaged:
        fake_wheel.response(steer_torque_op * A_steer_torque, speed)
        throttle_out = throttle_op * A_throttle
        brake_out = brake_op * A_brake
        steer_angle_out = fake_wheel.angle
        # print(steer_torque_op)
      # print(steer_angle_out)
      vc = carla.VehicleControl(throttle=throttle_out, steer=steer_angle_out / 3.14, brake=brake_out, reverse=in_reverse)
      vehicle.apply_control(vc)

    rk.keep_time()

if __name__ == "__main__":
  params = Params()
  params.delete("Offroad_ConnectivityNeeded")
  from selfdrive.version import terms_version, training_version
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put("CommunityFeaturesToggle", "1")
  params.put("CalibrationParams", '{"vanishing_point": [582.06, 442.78], "valid_blocks": 20}')

  # no carla, still run
  try:
    import carla
  except ImportError:
    print("WARNING: NO CARLA")
    while 1:
      time.sleep(1)
    
  from multiprocessing import Process, Queue
  q = Queue()
  p = Process(target=go, args=(q,))
  p.daemon = True
  p.start()

  if args.joystick:
    # start input poll for joystick
    from lib.manual_ctrl import wheel_poll_thread
    wheel_poll_thread(q)
  else:
    # start input poll for keyboard
    from lib.keyboard_ctrl import keyboard_poll_thread
    keyboard_poll_thread(q)

