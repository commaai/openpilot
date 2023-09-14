import signal
import threading

from multiprocessing import Process, Queue
from abc import ABC, abstractmethod

import cereal.messaging as messaging
from openpilot.common.params import Params

from openpilot.common.numpy_fast import clip
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.test.helpers import set_params_enabled
from openpilot.selfdrive.car.honda.values import CruiseButtons
from openpilot.tools.sim.lib.common import SimulatorState, World
from openpilot.tools.sim.lib.simulated_car import SimulatedCar
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors


class SimulatorBridge(ABC):
  TICKS_PER_FRAME = 5

  def __init__(self, arguments):
    set_params_enabled()
    self.params = Params()

    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    self.params.put("CalibrationParams", msg.to_bytes())

    self.dual_camera = arguments.dual_camera
    self.high_quality = arguments.high_quality

    self._exit_event = threading.Event()
    self._threads = []
    self._keep_alive = True
    self.started = False
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self._exit = threading.Event()
    self.simulator_state = SimulatorState()

  def _on_shutdown(self, signal, frame):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      self._run(q)
    finally:
      self.close()

  def close(self):
    self.started = False
    self._exit_event.set()

  def run(self, queue, retries=-1):
    bridge_p = Process(target=self.bridge_keep_alive, args=(queue, retries), daemon=True)
    bridge_p.start()
    return bridge_p

  @abstractmethod
  def spawn_world(self) -> World:
    pass

  def _run(self, q: Queue):
    self.world = world = self.spawn_world()
    self.simulated_car = SimulatedCar()
    self.simulated_sensors = SimulatedSensors(self.dual_camera)

    rk = Ratekeeper(100, print_delay_threshold=None)

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      world.tick()

    throttle_manual = steer_manual = brake_manual = 0.

    # loop
    while self._keep_alive:
      # 1. Read the throttle, steer and brake from op or manual controls
      # 2. Set instructions in Carla
      # 3. Send current carstate to op via can

      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0

      self.simulator_state.cruise_button = 0

      throttle_manual = steer_manual = brake_manual = 0.

      # --------------Step 1-------------------------------
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
        elif m[0] == "ignition":
          self.simulator_state.ignition = not self.simulator_state.ignition
        elif m[0] == "quit":
          break

      self.simulator_state.user_brake = brake_manual
      self.simulator_state.user_gas = throttle_manual

      steer_manual = steer_manual * -40

      self.simulated_car.update(self.simulator_state)
      self.simulated_sensors.update(self.simulator_state, self.world)

      is_openpilot_engaged = self.simulated_car.sm['controlsState'].active

      if is_openpilot_engaged:
        throttle_op = clip(self.simulated_car.sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = clip(-self.simulated_car.sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = self.simulated_car.sm['carControl'].actuators.steeringAngleDeg

      throttle_out = throttle_op if is_openpilot_engaged else throttle_manual
      brake_out = brake_op if is_openpilot_engaged else brake_manual
      steer_out = steer_op if is_openpilot_engaged else steer_manual

      world.apply_controls(steer_out, throttle_out, brake_out)
      world.read_sensors(self.simulator_state)

      if rk.frame % self.TICKS_PER_FRAME == 0:
        world.read_cameras()
        world.tick()

      self.started = True
