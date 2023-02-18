import math
from tools.sim.bridge.common import World, SimulatorBridge, Camerad, STEER_RATIO, W, H, gps_callback, imu_callback

class CarlaWorld(World):
  def __init__(self, world, vehicle, camerad):
    import carla  # pylint: disable=import-error
    self.world = world
    self.vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)
    self.vehicle = vehicle
    self.max_steer_angle: float = vehicle.get_physics_control().wheels[0].max_steer_angle
    self.camerad = camerad

  def apply_controls(self, steer_sim, throttle_out, brake_out, _rk):
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

class CarlaBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def spawn_objects(self):
    camerad = Camerad()
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

    return CarlaWorld(world, vehicle, camerad)