#!/usr/bin/env python3
import os
import time
import math
import atexit
import numpy as np
import threading
import random
import cereal.messaging as messaging
import argparse
from common.params import Params
from common.realtime import Ratekeeper
from lib.can import can_function, sendcan_function
import queue

parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
parser.add_argument('--autopilot', action='store_true')
args = parser.parse_args()

pm = messaging.PubMaster(['frame', 'sensorEvents', 'can'])

W,H = 1164, 874

def cam_callback(image):
  img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
  img = np.reshape(img, (H, W, 4))
  img = img[:, :, [0,1,2]].copy()

  dat = messaging.new_message()
  dat.init('frame')
  dat.frame = {
    "frameId": image.frame,
    "image": img.tostring(),
  }
  pm.send('frame', dat)

def imu_callback(imu):
  #print(imu, imu.accelerometer)

  dat = messaging.new_message()
  dat.init('sensorEvents', 2)
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
    dat = messaging.new_message()
    dat.init('health')
    dat.valid = True
    dat.health = {
      'ignitionLine': True,
      'hwType': "whitePanda",
      'controlsAllowed': True
    }
    pm.send('health', dat)
    rk.keep_time()

def fake_driver_monitoring():
  pm = messaging.PubMaster(['driverState'])
  while 1:
    dat = messaging.new_message()
    dat.init('driverState')
    dat.driverState.faceProb = 1.0
    pm.send('driverState', dat)
    time.sleep(0.1)

def go():
  import carla
  client = carla.Client("127.0.0.1", 2000)
  client.set_timeout(5.0)
  world = client.load_world('Town03')

  settings = world.get_settings()
  settings.fixed_delta_seconds = 0.05
  world.apply_settings(settings)

  weather = carla.WeatherParameters(
      cloudyness=0.0,
      precipitation=0.0,
      precipitation_deposits=0.0,
      wind_intensity=0.0,
      sun_azimuth_angle=0.0,
      sun_altitude_angle=0.0)
  world.set_weather(weather)

  blueprint_library = world.get_blueprint_library()
  """
  for blueprint in blueprint_library.filter('sensor.*'):
     print(blueprint.id)
  exit(0)
  """

  world_map = world.get_map()

  vehicle_bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))
  vehicle = world.spawn_actor(vehicle_bp, random.choice(world_map.get_spawn_points()))

  if args.autopilot:
    vehicle.set_autopilot(True)

  blueprint = blueprint_library.find('sensor.camera.rgb')
  blueprint.set_attribute('image_size_x', str(W))
  blueprint.set_attribute('image_size_y', str(H))
  blueprint.set_attribute('fov', '70')
  blueprint.set_attribute('sensor_tick', '0.05')
  transform = carla.Transform(carla.Location(x=0.8, z=1.45))
  camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
  camera.listen(cam_callback)

  # TODO: wait for carla 0.9.7
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
  rk = Ratekeeper(100)
  steer_angle = 0
  while 1:
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    can_function(pm, speed, steer_angle, rk.frame, rk.frame%500 == 499)
    if rk.frame%5 == 0:
      throttle, brake, steer = sendcan_function(sendcan)
      steer_angle += steer/10000.0 # torque
      vc = carla.VehicleControl(throttle=throttle, steer=steer_angle, brake=brake)
      vehicle.apply_control(vc)
      print(speed, steer_angle, vc)

    rk.keep_time()

if __name__ == "__main__":
  params = Params()
  params.delete("Offroad_ConnectivityNeeded")
  from selfdrive.version import terms_version, training_version
  params.put("HasAcceptedTerms", terms_version)
  params.put("CompletedTrainingVersion", training_version)
  params.put("CommunityFeaturesToggle", "1")

  threading.Thread(target=health_function).start()
  threading.Thread(target=fake_driver_monitoring).start()

  # no carla, still run
  try:
    import carla
  except ImportError:
    print("WARNING: NO CARLA")
    while 1:
      time.sleep(1)
    
  go()

