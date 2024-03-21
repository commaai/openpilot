import signal
import threading
import functools

from multiprocessing import Process, Queue, Value
from abc import ABC, abstractmethod

from openpilot.common.params import Params
from openpilot.common.numpy_fast import clip
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.test.helpers import set_params_enabled
from openpilot.selfdrive.car.honda.values import CruiseButtons
from openpilot.tools.sim.lib.common import SimulatorState, World
from openpilot.tools.sim.lib.simulated_car import SimulatedCar
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors


def rk_loop(function, hz, exit_event: threading.Event):
  rk = Ratekeeper(hz, None)
  while not exit_event.is_set():
    function()
    rk.keep_time()


class SimulatorBridge(ABC):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality):
    set_params_enabled()
    self.params = Params()

    self.rk = Ratekeeper(100, None)

    self.dual_camera = dual_camera
    self.high_quality = high_quality

    self._exit_event = threading.Event()
    self._threads = []
    self._keep_alive = True
    self.started = Value('i', False)
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self._exit = threading.Event()
    self.simulator_state = SimulatorState()

    self.world: World | None = None

    self.past_startup_engaged = False

  def _on_shutdown(self, signal, frame):
    self.shutdown()

  def shutdown(self):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      self._run(q)
    finally:
      self.close("bridge terminated")

  def close(self, reason):
    self.started.value = False
    self._exit_event.set()

    if self.world is not None:
      self.world.close(reason)

  def run(self, queue, retries=-1):
    bridge_p = Process(name="bridge", target=self.bridge_keep_alive, args=(queue, retries))
    bridge_p.start()
    return bridge_p

  def print_status(self):
    print(
    f"""
State:
Ignition: {self.simulator_state.ignition} Engaged: {self.simulator_state.is_engaged}
    """)

  @abstractmethod
  def spawn_world(self) -> World:
    pass

  def _run(self, q: Queue):
    self.world = self.spawn_world()

    self.simulated_car = SimulatedCar()
    self.simulated_sensors = SimulatedSensors(self.dual_camera)

    self.simulated_car_thread = threading.Thread(target=rk_loop, args=(functools.partial(self.simulated_car.update, self.simulator_state),
                                                                        100, self._exit_event))
    self.simulated_car_thread.start()

    self.simulated_camera_thread = threading.Thread(target=rk_loop, args=(functools.partial(self.simulated_sensors.send_camera_images, self.world),
                                                                        20, self._exit_event))
    self.simulated_camera_thread.start()

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      self.world.tick()

    while self._keep_alive:
      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0

      self.simulator_state.cruise_button = 0
      self.simulator_state.left_blinker = False
      self.simulator_state.right_blinker = False

      throttle_manual = steer_manual = brake_manual = 0.

      # Read manual controls
      if not q.empty():
        message = q.get()
        m = message.split('_')
        if m[0] == "steer":
          steer_manual = float(m[1])
        elif m[0] == "throttle":
          throttle_manual = float(m[1])
        elif m[0] == "brake":
          brake_manual = float(m[1])
        elif m[0] == "cruise":
          if m[1] == "down":
            self.simulator_state.cruise_button = CruiseButtons.DECEL_SET
          elif m[1] == "up":
            self.simulator_state.cruise_button = CruiseButtons.RES_ACCEL
          elif m[1] == "cancel":
            self.simulator_state.cruise_button = CruiseButtons.CANCEL
          elif m[1] == "main":
            self.simulator_state.cruise_button = CruiseButtons.MAIN
        elif m[0] == "blinker":
          if m[1] == "left":
            self.simulator_state.left_blinker = True
          elif m[1] == "right":
            self.simulator_state.right_blinker = True
        elif m[0] == "ignition":
          self.simulator_state.ignition = not self.simulator_state.ignition
        elif m[0] == "reset":
          self.world.reset()
        elif m[0] == "quit":
          break

      self.simulator_state.user_brake = brake_manual
      self.simulator_state.user_gas = throttle_manual
      self.simulator_state.user_torque = steer_manual * -10000

      steer_manual = steer_manual * -40

      # Update openpilot on current sensor state
      self.simulated_sensors.update(self.simulator_state, self.world)

      self.simulated_car.sm.update(0)
      controlsState = self.simulated_car.sm['controlsState']
      self.simulator_state.is_engaged = controlsState.active

      if self.simulator_state.is_engaged:
        throttle_op = clip(self.simulated_car.sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = clip(-self.simulated_car.sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = self.simulated_car.sm['carControl'].actuators.steeringAngleDeg

        self.past_startup_engaged = True
      elif not self.past_startup_engaged and controlsState.engageable:
        self.simulator_state.cruise_button = CruiseButtons.DECEL_SET # force engagement on startup

      throttle_out = throttle_op if self.simulator_state.is_engaged else throttle_manual
      brake_out = brake_op if self.simulator_state.is_engaged else brake_manual
      steer_out = steer_op if self.simulator_state.is_engaged else steer_manual

      self.world.apply_controls(steer_out, throttle_out, brake_out)
      self.world.read_state()
      self.world.read_sensors(self.simulator_state)

      if self.world.exit_event.is_set():
        self.shutdown()

      if self.rk.frame % self.TICKS_PER_FRAME == 0:
        self.world.tick()
        self.world.read_cameras()

      if self.rk.frame % 25 == 0:
        self.print_status()

      self.started.value = True

      self.rk.keep_time()
