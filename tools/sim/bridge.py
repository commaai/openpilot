#!/usr/bin/env python3
import argparse
import math
import os
import signal
import threading
import time
from multiprocessing import Process, Queue
from typing import Any
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import cereal.messaging as messaging
from cereal import log
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.basedir import BASEDIR
from common.numpy_fast import clip
from common.params import Params
from common.realtime import DT_DMON, Ratekeeper
from selfdrive.car.honda.values import CruiseButtons
from selfdrive.test.helpers import set_params_enabled
from tools.sim.lib.can import can_function

SCALE = 1
W, H = 1928 // SCALE, 1208 // SCALE
REPEAT_COUNTER = 5
PRINT_DECIMATION = 100
STEER_RATIO = 15.

pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'accelerometer', 'gyroscope', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl', 'controlsState'])

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--town', type=str, default='Town04_Opt')
  parser.add_argument('--spawn_point', dest='num_selected_spawn_point', type=int, default=16)
  parser.add_argument('--host', dest='host', type=str, default='127.0.0.1')
  parser.add_argument('--port', dest='port', type=int, default=2000)
  parser.add_argument('--simulator', dest='simulator', type=str, default='metadrive')
  parser.add_argument('--frames_per_tick', dest='frames_per_tick', type=int, default=None)

  return parser.parse_args(add_args)


class VehicleState:
  def __init__(self):
    self.speed = 0.0
    self.angle = 0.0
    self.bearing_deg = 0.0
    self.vel = None
    self.cruise_button = 0
    self.is_engaged = False
    self.ignition = True


def steer_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 0.5
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new



class Camerad:
  def __init__(self):
    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, W, H)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, W, H)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)
    cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "

    kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(cl_arg)
      self.krnl = prg.rgb_to_nv12
    self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4

  def cam_callback_road(self, image):
    yuv = self.img_to_yuv(image)
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_callback_wide_road(self, image):
    yuv = self.img_to_yuv(image)
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  # Returns: yuv bytes
  def img_to_yuv(self, img):
    assert img.shape == (H, W, 3), f"{img.shape}"
    rgb = np.reshape(img, (H, W * 3))
    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (np.int32(self.Wdiv4), np.int32(self.Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), rgb.size // 2)
    return yuv.data.tobytes()
  
  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = int(frame_id * 0.05 * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)

    dat = messaging.new_message(pub_type)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    pm.send(pub_type, dat)

def imu_callback(imu, vehicle_state):
  # send 5x since 'sensor_tick' doesn't seem to work. limited by the world tick?
  for _ in range(5):
    vehicle_state.bearing_deg = math.degrees(imu.compass)
    dat = messaging.new_message('accelerometer')
    dat.accelerometer.sensor = 4
    dat.accelerometer.type = 0x10
    dat.accelerometer.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
    pm.send('accelerometer', dat)

    # copied these numbers from locationd
    dat = messaging.new_message('gyroscope')
    dat.gyroscope.sensor = 5
    dat.gyroscope.type = 0x10
    dat.gyroscope.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
    pm.send('gyroscope', dat)
    time.sleep(0.01)


def panda_state_function(vs: VehicleState, exit_event: threading.Event):
  pm = messaging.PubMaster(['pandaStates'])
  while not exit_event.is_set():
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': vs.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec',
    }
    pm.send('pandaStates', dat)
    time.sleep(0.5)


def peripheral_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['peripheralState'])
  while not exit_event.is_set():
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    # fake peripheral state data
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    pm.send('peripheralState', dat)
    time.sleep(0.5)


def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')

  # transform vel from carla to NED
  # north is -Y in CARLA
  velNED = [
    -vehicle_state.vel.y,  # north/south component of NED is negative when moving south
    vehicle_state.vel.x,  # positive when moving east, which is x in carla
    vehicle_state.vel.z,
  ]

  dat.gpsLocationExternal = {
    "unixTimestampMillis": int(time.time() * 1000),
    "flags": 1,  # valid fix
    "accuracy": 1.0,
    "verticalAccuracy": 1.0,
    "speedAccuracy": 0.1,
    "bearingAccuracyDeg": 0.1,
    "vNED": velNED,
    "bearingDeg": vehicle_state.bearing_deg,
    "latitude": gps.latitude,
    "longitude": gps.longitude,
    "altitude": gps.altitude,
    "speed": vehicle_state.speed,
    "source": log.GpsLocationData.SensorSource.ublox,
  }

  pm.send('gpsLocationExternal', dat)


def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverStateV2', 'driverMonitoringState'])
  while not exit_event.is_set():
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceOrientation = [0.0, 1.0, 0.0]
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    pm.send('driverStateV2', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)

    time.sleep(DT_DMON)


def can_function_runner(vs: VehicleState, exit_event: threading.Event):
  i = 0
  while not exit_event.is_set():
    can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged)
    time.sleep(0.01)
    i += 1

class World(ABC):
  @abstractmethod
  def apply_controls(self, steer_sim, throttle_out, brake_out, frame: int):
    pass
  @abstractmethod
  def get_velocity(self):
    pass
  @abstractmethod
  def get_speed(self) -> float:
    pass
  @abstractmethod
  def get_steer_correction(self) -> float:
    pass
  @abstractmethod
  def tick(self):
    pass

class CarlaWorld(World):
  def __init__(self, world, vc, vehicle, camerad):
    self.world = world
    self.vc = vc
    self.vehicle = vehicle
    self.max_steer_angle: float = vehicle.get_physics_control().wheels[0].max_steer_angle
    self.camerad = camerad

  def apply_controls(self, steer_sim, throttle_out, brake_out, _frame):
    self.vc.throttle = throttle_out / 0.6
    self.vc.steer = steer_sim
    self.vc.brake = brake_out
    self.vehicle.apply_control(self.vc)

  def get_velocity(self):
    self.vel = self.vehicle.get_velocity()
    return self.vel

  def get_speed(self) -> float:
    return math.sqrt(self.vel.x ** 2 + self.vel.y ** 2 + self.vel.z ** 2)  # in m/s

  def get_steer_correction(self) -> float:
    return self.max_steer_angle * STEER_RATIO * -1  
  
  def tick(self):
    self.world.tick()

class MetaDriveWorld(World):
  def __init__(self, env, frames_per_tick: float):
    self.env = env
    self.speed = 0.0
    self.yuv = None
    self.camerad = Camerad()
    self.frames_per_tick = frames_per_tick

  def apply_controls(self, steer_sim, throttle_out, brake_out, frame: int):
    vc = [0.0, 0.0]
    vc[0] = steer_sim * -1
    if throttle_out:
      vc[1] = throttle_out * 10
    else:
      vc[1] = -brake_out

    if frame % self.frames_per_tick == 0 or self.yuv is None:
      if frame % (self.frames_per_tick * 2) == 0:
        o, _, _, _ = self.env.step(vc)
        self.speed = o["state"][3] * self.frames_per_tick * 2 # empirically derived
      img = self.env.vehicle.image_sensors["rgb_wide"].get_pixels_array(self.env.vehicle, False)
      self.yuv = self.camerad.img_to_yuv(img)

    if frame % (self.frames_per_tick * 2) == 0:
      self.camerad.cam_send_yuv_wide_road(self.yuv)

  def get_velocity(self):
    return None

  def get_speed(self) -> float:
    return self.speed

  def get_steer_correction(self) -> float:
    max_steer_angle = 75 / self.frames_per_tick
    return max_steer_angle * STEER_RATIO * -1
  
  def tick(self):
    pass


class SimulatorBridge(ABC):
  FRAMES_PER_TICK = 5
  def __init__(self, arguments):
    if arguments.frames_per_tick:
      self.frames_per_tick = arguments.frames_per_tick
    else:
      self.frames_per_tick = self.FRAMES_PER_TICK
    print(f"Running at {self.frames_per_tick} frames per tick")
    set_params_enabled()
    self.params = Params()

    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    self.params.put("CalibrationParams", msg.to_bytes())
    if not arguments.dual_camera:
      print("Dual Camera disabled")
    self.params.put_bool("WideCameraOnly", not arguments.dual_camera)
    self.params.put_bool("DisengageOnAccelerator", True)

    self._args = arguments
    self._simulation_objects = []
    self._exit_event = threading.Event()
    self._threads = []
    self._keep_alive = True
    self.started = False
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self._exit = threading.Event()
  
  def _on_shutdown(self, signal, frame):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      while self._keep_alive:
        try:
          self._run(q)
          break
        except RuntimeError as e:
          self.close()
          if retries == 0:
            raise

          # Reset for another try
          self._simulation_objects = []
          self._threads = []
          self._exit_event = threading.Event()

          retries -= 1
          if retries <= -1:
            print(f"Restarting bridge. Error: {e} ")
          else:
            print(f"Restarting bridge. Retries left {retries}. Error: {e} ")
    finally:
      # Clean up resources in the opposite order they were created.
      self.close()
  
  def _run(self, q: Queue):
    self._vehicle_state = VehicleState()
    world = self.spawn_objects()

    # launch fake car threads
    self._threads.append(threading.Thread(target=panda_state_function, args=(self._vehicle_state, self._exit_event,)))
    self._threads.append(threading.Thread(target=peripheral_state_function, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=fake_driver_monitoring, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=can_function_runner, args=(self._vehicle_state, self._exit_event,)))
    for t in self._threads:
      t.start()

    # init
    throttle_ease_out_counter = REPEAT_COUNTER
    brake_ease_out_counter = REPEAT_COUNTER
    steer_ease_out_counter = REPEAT_COUNTER

    is_openpilot_engaged = False
    throttle_out = steer_out = brake_out = 0.
    throttle_op = steer_op = brake_op = 0.
    throttle_manual = steer_manual = brake_manual = 0.

    old_steer = old_brake = old_throttle = 0.
    throttle_manual_multiplier = 0.7  # keyboard signal is always 1
    brake_manual_multiplier = 0.7  # keyboard signal is always 1
    steer_manual_multiplier = 45 * STEER_RATIO  # keyboard signal is always 1

    # loop
    rk = Ratekeeper(100, print_delay_threshold=0.05)

    while self._keep_alive:
      # 1. Read the throttle, steer and brake from op or manual controls
      # 2. Set instructions in Carla
      # 3. Send current carstate to op via can

      cruise_button = 0
      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0
      throttle_manual = steer_manual = brake_manual = 0.0

      # --------------Step 1-------------------------------
      if not q.empty():
        message = q.get()
        m = message.split('_')
        if m[0] == "steer":
          steer_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "throttle":
          throttle_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "brake":
          brake_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "reverse":
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
        elif m[0] == "cruise":
          if m[1] == "down":
            cruise_button = CruiseButtons.DECEL_SET
            is_openpilot_engaged = True
          elif m[1] == "up":
            cruise_button = CruiseButtons.RES_ACCEL
            is_openpilot_engaged = True
          elif m[1] == "cancel":
            cruise_button = CruiseButtons.CANCEL
            is_openpilot_engaged = False
        elif m[0] == "ignition":
          self._vehicle_state.ignition = not self._vehicle_state.ignition
        elif m[0] == "quit":
          break

        throttle_out = throttle_manual * throttle_manual_multiplier
        steer_out = steer_manual * steer_manual_multiplier
        brake_out = brake_manual * brake_manual_multiplier

        old_steer = steer_out
        old_throttle = throttle_out
        old_brake = brake_out

      if is_openpilot_engaged:
        sm.update(0)

        # TODO gas and brake is deprecated
        throttle_op = clip(sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = clip(-sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = sm['carControl'].actuators.steeringAngleDeg

        throttle_out = throttle_op
        steer_out = steer_op
        brake_out = brake_op

        steer_out = steer_rate_limit(old_steer, steer_out)
        old_steer = steer_out

      else:
        if throttle_out == 0 and old_throttle > 0:
          if throttle_ease_out_counter > 0:
            throttle_out = old_throttle
            throttle_ease_out_counter += -1
          else:
            throttle_ease_out_counter = REPEAT_COUNTER
            old_throttle = 0

        if brake_out == 0 and old_brake > 0:
          if brake_ease_out_counter > 0:
            brake_out = old_brake
            brake_ease_out_counter += -1
          else:
            brake_ease_out_counter = REPEAT_COUNTER
            old_brake = 0

        if steer_out == 0 and old_steer != 0:
          if steer_ease_out_counter > 0:
            steer_out = old_steer
            steer_ease_out_counter += -1
          else:
            steer_ease_out_counter = REPEAT_COUNTER
            old_steer = 0

      # --------------Step 2-------------------------------
      steer_correction = world.get_steer_correction()
      steer_sim = steer_out / steer_correction

      steer_sim = np.clip(steer_sim, -1, 1)
      steer_out = steer_sim * steer_correction
      old_steer = steer_sim * steer_correction

      world.apply_controls(steer_sim, throttle_out, brake_out, rk.frame)

      # --------------Step 3-------------------------------
      self._vehicle_state.speed = world.get_speed()
      self._vehicle_state.vel = world.get_velocity()
      self._vehicle_state.angle = steer_out
      self._vehicle_state.cruise_button = cruise_button
      self._vehicle_state.is_engaged = is_openpilot_engaged

      if rk.frame % PRINT_DECIMATION == 0:
        print("frame: ", "engaged:", is_openpilot_engaged, "; throttle: ", round(throttle_out, 3), "; steer(deg): ",
              round(steer_out, 3), "; brake: ", round(brake_out, 3))

      if rk.frame % self.FRAMES_PER_TICK == 0:
        world.tick()
      rk.keep_time()
      self.started = True

  def close(self):
    self.started = False
    self._exit_event.set()

    for s in self._simulation_objects:
      try:
        s.destroy()
      except Exception as e:
        print("Failed to destroy carla object", e)
    for t in reversed(self._threads):
      t.join()

  def run(self, queue, retries=-1):
    bridge_p = Process(target=self.bridge_keep_alive, args=(queue, retries), daemon=True)
    bridge_p.start()
    return bridge_p
  
  # Must return object of class `World`, after spawning objects into that world
  @abstractmethod
  def spawn_objects(self):
    pass

class CarlaBridge(SimulatorBridge):
  FRAMES_PER_TICK = 5

  def spawn_objects(self):
    camerad = Camerad()
    # Simulator specific imports go here
    import carla  # pylint: disable=import-error
    def connect_carla_client(host: str, port: int):
      client = carla.Client(host, port)
      client.set_timeout(5)
      return client

    client = connect_carla_client(self._args.host, self._args.port)
    world = client.load_world(self._args.town)

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearSunset)

    if not self._args.high_quality:
      world.unload_map_layer(carla.MapLayer.Foliage)
      world.unload_map_layer(carla.MapLayer.Buildings)
      world.unload_map_layer(carla.MapLayer.ParkedVehicles)
      world.unload_map_layer(carla.MapLayer.Props)
      world.unload_map_layer(carla.MapLayer.StreetLights)
      world.unload_map_layer(carla.MapLayer.Particles)

    blueprint_library = world.get_blueprint_library()

    world_map = world.get_map()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_points = world_map.get_spawn_points()
    assert len(spawn_points) > self._args.num_selected_spawn_point, f'''No spawn point {self._args.num_selected_spawn_point}, try a value between 0 and
      {len(spawn_points)} for this town.'''
    spawn_point = spawn_points[self._args.num_selected_spawn_point]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    self._simulation_objects.append(vehicle)

    # make tires less slippery
    # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
    physics_control = vehicle.get_physics_control()
    physics_control.mass = 2326
    # physics_control.wheels = [wheel_control]*4
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    vehicle.apply_physics_control(physics_control)

    transform = carla.Transform(carla.Location(x=0.8, z=1.13))

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      if not self._args.high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
      camera.listen(callback)
      return camera

    if self._args.dual_camera:
      road_camera = create_camera(fov=40, callback=camerad.cam_callback_road)
      self._simulation_objects.append(road_camera)

    road_wide_camera = create_camera(fov=120, callback=camerad.cam_callback_wide_road)  # fov bigger than 120 shows unwanted artifacts
    self._simulation_objects.append(road_wide_camera)

    # re-enable IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    imu.listen(lambda imu: imu_callback(imu, self._vehicle_state))

    gps_bp = blueprint_library.find('sensor.other.gnss')
    gps = world.spawn_actor(gps_bp, transform, attach_to=vehicle)
    gps.listen(lambda gps: gps_callback(gps, self._vehicle_state))
    self.params.put_bool("UbloxAvailable", True)
    self._simulation_objects.extend([imu, gps])

    vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)

    return CarlaWorld(world, vc, vehicle, camerad)

class MetaDriveBridge(SimulatorBridge):
  FRAMES_PER_TICK = 15

  def __init__(self, args):
    if args.dual_camera:
      warnings.warn("Dual camera not supported in MetaDrive simulator for performance reasons")
      args.dual_camera = False
    super(MetaDriveBridge, self).__init__(args)

  def spawn_objects(self):
    import metadrive  # noqa: F401 pylint: disable=W0611
    import gym
    env = gym.make('MetaDrive-10env-v0', config=dict(offscreen_render=True))
  
    # from metadrive.utils.space import VehicleParameterSpace
    # from metadrive.component.vehicle.vehicle_type import DefaultVehicle
    # max_engine_force = VehicleParameterSpace.DEFAULT_VEHICLE["max_engine_force"]
    # max_engine_force._replace(max=max_engine_force.max * 10)
    # max_engine_force._replace(min=max_engine_force.min * 10)

    # config = dict(
    #   # camera_dist=3.0,
    #   # camera_height=1.0,
    #   # use_render=True,
    #   vehicle_config=dict(
    #     # enable_reverse=True,
    #     # image_source="rgb_camera",
    #     # rgb_camera=(0,0)
    #   ),
    #   offscreen_render=True,
    # )
  
    env.reset()
    from metadrive.constants import CamMask
    from metadrive.component.vehicle_module.base_camera import BaseCamera
    from metadrive.engine.engine_utils import engine_initialized
    from metadrive.engine.core.image_buffer import ImageBuffer
    class RGBCameraWide(BaseCamera):
      # shape(dim_1, dim_2)
      BUFFER_W = W  # dim 1
      BUFFER_H = H  # dim 2
      CAM_MASK = CamMask.RgbCam

      def __init__(self):
        assert engine_initialized(), "You should initialize engine before adding camera to vehicle"
        self.BUFFER_W, self.BUFFER_H = W, H
        super(RGBCameraWide, self).__init__()
        print("SELF", self.BUFFER_W, self.BUFFER_H)
        cam = self.get_cam()
        lens = self.get_lens()
        cam.lookAt(0, 2.4, 1.3)
        cam.setHpr(0, 0.8, 0)
        lens.setFov(160)
        lens.setAspectRatio(1.15)
  
    env.vehicle.add_image_sensor("rgb_wide", RGBCameraWide())

    # Monkey patch `get_rgb_array` since it involves unnecessary copies
    def patch_get_rgb_array(self):
      if self.engine.episode_step <= 1:
        self.engine.graphicsEngine.renderFrame()
      origin_img = self.cam.node().getDisplayRegion(0).getScreenshot()
      img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
      img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
      img = img[::-1]
      img = img[..., :-1]
      return img

    setattr(ImageBuffer, "get_rgb_array", patch_get_rgb_array)

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      env.step([1.0, 0.0])

    return MetaDriveWorld(env, self.FRAMES_PER_TICK)

if __name__ == "__main__":
  q: Any = Queue()
  args = parse_args()
  print("HELLO WORLD")

  try:
    simulator_bridge: SimulatorBridge
    if args.simulator == "metadrive":
      print("STARTING METADRIVE BRIDGE")
      simulator_bridge = MetaDriveBridge(args)
    elif args.simulator == "carla":
      simulator_bridge = CarlaBridge(args)
    else:
      raise AssertionError("simulator type not supported")
    p = simulator_bridge.run(q)

    if args.joystick:
      # start input poll for joystick
      from tools.sim.lib.manual_ctrl import wheel_poll_thread

      wheel_poll_thread(q)
    else:
      # start input poll for keyboard
      from tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

      keyboard_poll_thread(q)
    p.join()

  finally:
    # Try cleaning up the wide camera param
    # in case users want to use replay after
    Params().remove("WideCameraOnly")
