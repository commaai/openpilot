import numpy as np

from openpilot.common.params import Params
from openpilot.tools.sim.lib.common import SimulatorState, vec3
from openpilot.tools.sim.bridge.common import World, SimulatorBridge
from openpilot.tools.sim.lib.camerad import W, H


class CarlaWorld(World):
  def __init__(self, client, high_quality, dual_camera, num_selected_spawn_point, town):
    super().__init__(dual_camera)
    import carla

    low_quality_layers = carla.MapLayer(carla.MapLayer.Ground | carla.MapLayer.Walls | carla.MapLayer.Decals)

    layers = carla.MapLayer.All if high_quality else low_quality_layers

    world = client.load_world(town, map_layers=layers)

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.01
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearSunset)

    self.world = world
    world_map = world.get_map()

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_points = world_map.get_spawn_points()
    assert len(spawn_points) > num_selected_spawn_point, \
      f'''No spawn point {num_selected_spawn_point}, try a value between 0 and {len(spawn_points)} for this town.'''
    self.spawn_point = spawn_points[num_selected_spawn_point]
    self.vehicle = world.spawn_actor(vehicle_bp, self.spawn_point)

    physics_control = self.vehicle.get_physics_control()
    physics_control.mass = 2326
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    self.vehicle.apply_physics_control(physics_control)

    self.vc: carla.VehicleControl = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)
    self.max_steer_angle: float = self.vehicle.get_physics_control().wheels[0].max_steer_angle
    self.params = Params()

    self.steer_ratio = 15

    self.carla_objects = []

    transform = carla.Transform(carla.Location(x=0.8, z=1.13))

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      blueprint.set_attribute('sensor_tick', str(1/20))
      if not high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = world.spawn_actor(blueprint, transform, attach_to=self.vehicle)
      camera.listen(callback)
      return camera

    self.road_camera = create_camera(fov=40, callback=self.cam_callback_road)
    if dual_camera:
      self.road_wide_camera = create_camera(fov=120, callback=self.cam_callback_wide_road)  # fov bigger than 120 shows unwanted artifacts
    else:
      self.road_wide_camera = None

    # re-enable IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    self.imu = world.spawn_actor(imu_bp, transform, attach_to=self.vehicle)

    gps_bp = blueprint_library.find('sensor.other.gnss')
    self.gps = world.spawn_actor(gps_bp, transform, attach_to=self.vehicle)
    self.params.put_bool("UbloxAvailable", True)

    self.carla_objects = [self.imu, self.gps, self.road_camera, self.road_wide_camera, self.vehicle]

  def close(self):
    for s in self.carla_objects:
      if s is not None:
        try:
          s.destroy()
        except Exception as e:
          print("Failed to destroy carla object", e)

  def carla_image_to_rgb(self, image):
    rgb = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    rgb = np.reshape(rgb, (H, W, 4))
    return np.ascontiguousarray(rgb[:, :, [0, 1, 2]])

  def cam_callback_road(self, image):
    with self.image_lock:
      self.road_image = self.carla_image_to_rgb(image)

  def cam_callback_wide_road(self, image):
    with self.image_lock:
      self.wide_road_image = self.carla_image_to_rgb(image)

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    self.vc.throttle = throttle_out

    steer_carla = steer_angle * -1 / (self.max_steer_angle * self.steer_ratio)
    steer_carla = np.clip(steer_carla, -1, 1)

    self.vc.steer = steer_carla
    self.vc.brake = brake_out
    self.vehicle.apply_control(self.vc)

  def read_sensors(self, simulator_state: SimulatorState):
    simulator_state.imu.bearing = self.imu.get_transform().rotation.yaw

    simulator_state.imu.accelerometer = vec3(
      self.imu.get_acceleration().x,
      self.imu.get_acceleration().y,
      self.imu.get_acceleration().z
    )

    simulator_state.imu.gyroscope = vec3(
      self.imu.get_angular_velocity().x,
      self.imu.get_angular_velocity().y,
      self.imu.get_angular_velocity().z
    )

    simulator_state.gps.from_xy([self.vehicle.get_location().x, self.vehicle.get_location().y])

    simulator_state.velocity = self.vehicle.get_velocity()
    simulator_state.valid = True
    simulator_state.steering_angle = self.vc.steer * self.max_steer_angle

  def read_cameras(self):
    pass # cameras are read within a callback for carla

  def tick(self):
    self.world.tick()

  def reset(self):
    import carla

    self.vehicle.set_transform(self.spawn_point)
    self.vehicle.set_target_velocity(carla.Vector3D())


class CarlaBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, arguments):
    super().__init__(arguments)
    self.host = arguments.host
    self.port = arguments.port
    self.town = arguments.town
    self.num_selected_spawn_point = arguments.num_selected_spawn_point

  def spawn_world(self):
    import carla
    client = carla.Client(self.host, self.port)
    client.set_timeout(5)

    return CarlaWorld(client, high_quality=self.high_quality, dual_camera=self.dual_camera,
                      num_selected_spawn_point=self.num_selected_spawn_point, town=self.town)