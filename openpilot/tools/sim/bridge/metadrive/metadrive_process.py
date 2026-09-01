import math
import os
import time
import numpy as np

from collections import namedtuple
from panda3d.core import Vec3
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

from openpilot.common.realtime import Ratekeeper

from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0,0)
METADRIVE_STEER_RATIO = 8
CAMERA_WHITE_VALUE = 220
CAMERA_WHITE_FRACTION = 0.7


metadrive_simulation_state = namedtuple("metadrive_simulation_state", ["running", "done", "done_info"])
metadrive_vehicle_state = namedtuple("metadrive_vehicle_state", ["velocity", "position", "bearing", "steering_angle"])

def apply_metadrive_patches(arrive_dest_done=True, out_of_road_done=True):
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
      sensor = cls(*args, self, cuda=True)
    else:
      sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, *args, **kwargs):
    return self.state

  ImageObservation.observe = observe_patched

  # disable destination, we want to loop forever
  def arrive_destination_patch(self, *args, **kwargs):
    return False

  if not arrive_dest_done:
    MetaDriveEnv._is_arrive_destination = arrive_destination_patch

  # MetaDrive 0.4.2.3 has no out_of_road_done configuration option. Disable
  # its instantaneous check here and let the bridge's debounced lane check
  # decide whether the vehicle has actually left the road.
  if not out_of_road_done:
    def out_of_road_patch(self, vehicle):
      return False

    MetaDriveEnv._is_out_of_road = out_of_road_patch

def metadrive_process(dual_camera: bool, config: dict, camera_array, wide_camera_array, image_lock,
                      controls_recv: Connection, simulation_state_send: Connection, vehicle_state_send: Connection,
                      exit_event, op_engaged, test_duration, test_run):
  from openpilot.tools.sim.bridge.metadrive.ci_render_patches import apply_ci_render_patches
  apply_ci_render_patches()

  arrive_dest_done = config.pop("arrive_dest_done", True)
  out_of_road_done = config.pop("out_of_road_done", True)
  apply_metadrive_patches(arrive_dest_done, out_of_road_done)

  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
  if dual_camera:
    assert wide_camera_array is not None
    wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = MetaDriveEnv(config)
  physics_step = config.get("physics_world_step_size", 0.05)
  out_of_lane_debounce = max(1, round(1.0 / physics_step))

  def get_current_lane_info(vehicle):
    _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
    lane_idx = lane_info[2] if lane_info is not None else None
    return lane_idx, on_lane

  def reset():
    env.reset()
    env.vehicle.config["max_speed_km_h"] = float(os.environ.get("METADRIVE_MAX_SPEED_KMH", "1000"))
    lane_idx_prev, _ = get_current_lane_info(env.vehicle)

    simulation_state = metadrive_simulation_state(
      running=True,
      done=False,
      done_info=None,
    )
    simulation_state_send.send(simulation_state)

    return lane_idx_prev

  lane_idx_prev = reset()
  start_time = None
  out_of_lane_streak = 0

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get() # convert cupy array to numpy
    if img.shape[:2] != (H, W):
      y_scale, x_scale = H // img.shape[0], W // img.shape[1]
      if (img.shape[0] * y_scale, img.shape[1] * x_scale) != (H, W):
        raise ValueError(f"METADRIVE_RENDER_SCALE must produce a resolution that evenly divides {(W, H)}")
      img = img.repeat(y_scale, axis=0).repeat(x_scale, axis=1)
    # llvmpipe can occasionally return the clear color while the tagged
    # terrain card is being repositioned. Keep one valid frame per camera so a
    # transient render miss cannot become a white model input.
    camera_sample = img[H // 2::16, ::16]
    invalid_frame = np.mean(np.all(camera_sample > CAMERA_WHITE_VALUE, axis=2)) > CAMERA_WHITE_FRACTION
    if invalid_frame and cam in last_valid_camera:
      return last_valid_camera[cam]
    if not invalid_frame:
      last_valid_camera[cam] = img.copy()
    return img

  rk = Ratekeeper(100, None)

  vc = [0,0]
  rendered_frames = 0
  render_start = time.monotonic()
  invalid_camera_streak = 0
  max_invalid_camera_streak = 0
  invalid_camera_frames = 0
  last_valid_camera = {}

  while not exit_event.is_set():
    vehicle_state = metadrive_vehicle_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
    )
    vehicle_state_send.send(vehicle_state)

    if controls_recv.poll(0):
      while controls_recv.poll(0):
        steer_angle, gas, should_reset = controls_recv.recv()

      steer_metadrive = steer_angle / (env.vehicle.MAX_STEERING * METADRIVE_STEER_RATIO)
      steer_metadrive = np.clip(steer_metadrive, -1, 1)

      vc = [steer_metadrive, gas]

      if should_reset:
        lane_idx_prev = reset()
        start_time = None
        out_of_lane_streak = 0

    is_engaged = op_engaged.is_set()
    if is_engaged and start_time is None:
      start_time = time.monotonic()

    if rk.frame % 5 == 0:
      _, _, terminated, _, _ = env.step(vc)
      timeout = True if start_time is not None and time.monotonic() - start_time >= test_duration else False
      lane_idx_curr, on_lane = get_current_lane_info(env.vehicle)
      lane_changed = lane_idx_curr != lane_idx_prev
      if is_engaged and start_time is not None and not on_lane:
        out_of_lane_streak += 1
      else:
        out_of_lane_streak = 0
      # A lane-index transition is a valid lane change when MetaDrive still
      # considers the vehicle on a lane. Only sustained off-road detection is
      # a driving failure; this also filters one-frame lane-detector flicker.
      out_of_lane = out_of_lane_streak >= out_of_lane_debounce
      lane_idx_before = lane_idx_prev
      lane_idx_prev = lane_idx_curr

      if terminated or ((out_of_lane or timeout) and test_run):
        camera_invalid = invalid_camera_frames > 0
        camera_health = "".join((
          "metadrive camera health: ",
          f"invalid_frames={invalid_camera_frames} max_invalid_streak={max_invalid_camera_streak} ",
          f"invalid={camera_invalid}",
        ))
        print(camera_health, flush=True)
        if terminated or out_of_lane:
          diagnostic = "".join((
            "metadrive termination diagnostic: ",
            f"terminated={terminated} lane_changed={lane_changed} on_lane={on_lane} ",
            f"lane_before={lane_idx_before} lane_now={lane_idx_curr} ",
            f"position={tuple(float(x) for x in env.vehicle.position)} ",
            f"speed_km_h={float(env.vehicle.speed_km_h):.2f} ",
            f"heading_deg={math.degrees(env.vehicle.heading_theta):.2f} ",
            f"steering={float(env.vehicle.steering):.4f} controls={vc}",
          ))
          print(diagnostic, flush=True)
        if terminated:
          done_result = env.done_function("default_agent")
        elif out_of_lane:
          done_result = (True, {"out_of_lane" : True, "camera_invalid": camera_invalid})
        elif timeout:
          done_result = (True, {"timeout" : True, "camera_invalid": camera_invalid})

        simulation_state = metadrive_simulation_state(
          running=False,
          done=done_result[0],
          done_info=done_result[1],
        )
        simulation_state_send.send(simulation_state)

      if dual_camera:
        wide_road_image[...] = get_cam_as_rgb("rgb_wide")
      road_image[...] = get_cam_as_rgb("rgb_road")
      # A broken terrain render can cover the lower camera view with solid
      # white for multiple frames. That makes the model steer on corrupt input,
      # so make it an explicit test failure instead of a flaky road departure.
      camera_sample = road_image[H // 2::16, ::16]
      invalid_camera_frame = (op_engaged.is_set() and
                              np.mean(np.all(camera_sample > CAMERA_WHITE_VALUE, axis=2)) > CAMERA_WHITE_FRACTION)
      invalid_camera_frames += int(invalid_camera_frame)
      invalid_camera_streak = invalid_camera_streak + 1 if invalid_camera_frame else 0
      max_invalid_camera_streak = max(max_invalid_camera_streak, invalid_camera_streak)
      image_lock.release()

      rendered_frames += 1
      if os.environ.get("METADRIVE_REPORT_FPS") and rendered_frames % 100 == 0:
        now = time.monotonic()
        print(f"metadrive render fps: {100 / (now - render_start):.1f} (target 20)", flush=True)
        render_start = now

    rk.keep_time()

