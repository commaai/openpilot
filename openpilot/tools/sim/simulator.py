import os
import signal
import threading
import time
import traceback
import numpy as np

from collections import deque
from multiprocessing import Process, Queue
from opendbc.car.honda.values import CruiseButtons
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, DT_MDL, Ratekeeper
from openpilot.selfdrive.test.helpers import set_params_enabled
from openpilot.tools.sim.car import Driver, SimulatedCar
from openpilot.tools.sim.route import RouteData
from openpilot.tools.sim.sensors import SimulatedSensors
from openpilot.tools.sim.world import LogWorld

STATUS_HZ = 5
MAX_CATCH_UP = 0.5  # s
MIN_FRAME_SPACING = 0.8  # of a frame
# terminals repeat a held key at ~30 Hz, so a manual input stays on this long after each press. it's
# also long enough for a button press to make it into a CAN message, which go out on their own clock
HOLD = 0.1  # s
MANUAL_INPUTS = {"gas": ("gas", 1.0), "brake": ("brake", 1.0), "steer_left": ("steer", -0.15), "steer_right": ("steer", 0.15)}
CRUISE_BUTTONS = {"cruise_up": CruiseButtons.RES_ACCEL, "cruise_down": CruiseButtons.DECEL_SET, "cruise_cancel": CruiseButtons.CANCEL}


def start_simulator(route: str, start_sec: float, cmd_q: Queue, status_q: Queue, log_path: str | None = None) -> Process:
  """Runs the simulated car, its sensors and cameras for openpilot in a new process. It takes commands (see Simulator.handle) from cmd_q,
  and puts ("status", dict) at STATUS_HZ, ("episode_end", dict) when the episode ends and ("error", str) if it crashes on status_q."""
  p = Process(name="simulator", target=_simulator_main, args=(route, start_sec, cmd_q, status_q, log_path))
  p.start()
  return p


def _simulator_main(route, start_sec, cmd_q, status_q, log_path):
  # the parent shuts us down with terminate(), and may have stopped reading status by then
  for sig in (signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, signal.SIG_IGN)
  signal.signal(signal.SIGTERM, signal.SIG_DFL)
  status_q.cancel_join_thread()
  if log_path is not None:
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
  try:
    sim = Simulator(route, start_sec, cmd_q, status_q)
    signal.signal(signal.SIGTERM, lambda *_: sim.shutdown())
    sim.run()
  except Exception:
    traceback.print_exc()
    status_q.put(("error", traceback.format_exc()))


class Simulator:
  def __init__(self, route: str, start_sec: float, cmd_q: Queue, status_q: Queue):
    self.cmd_q = cmd_q
    self.status_q = status_q
    self.keep_alive = True
    signal.signal(signal.SIGTERM, lambda *_: self.shutdown())

    set_params_enabled()
    params = Params()
    params.put_bool("AlphaLongitudinalEnabled", True, block=True)
    params.put_bool("UbloxAvailable", True, block=True)  # the sim's GPS is gpsLocationExternal
    route_data = RouteData(route)
    # written before the car turns on, so calibrationd starts out with the route's mounting
    params.put("CalibrationParams", route_data.calib_msg, block=True)

    start_frame = int(np.searchsorted(route_data.t, route_data.t[0] + start_sec))
    self.world = LogWorld(route_data, start_frame=start_frame)
    self.car = SimulatedCar()
    self.sensors = SimulatedSensors()

    self.driver = Driver()
    self.repeated: deque[bool] = deque(maxlen=20)  # whether each of the last second of camera frames repeated the one before
    self.engaged = False
    self.manual = {"gas": 0., "brake": 0., "steer": 0.}
    self.manual_until = 0.
    self.cruise_button = 0
    self.cruise_button_until = 0.
    self.past_startup_engaged = False
    self.startup_button_prev = True

  def shutdown(self):
    self.keep_alive = False

  def handle(self, cmd: str):
    if cmd in MANUAL_INPUTS:
      name, value = MANUAL_INPUTS[cmd]
      self.manual[name] = value
      self.manual_until = time.monotonic() + HOLD
    elif cmd in CRUISE_BUTTONS:
      self.cruise_button = CRUISE_BUTTONS[cmd]
      self.cruise_button_until = time.monotonic() + HOLD
    elif cmd == "blinker_left":
      self.driver.left_blinker = not self.driver.left_blinker
      self.driver.right_blinker = False
    elif cmd == "blinker_right":
      self.driver.right_blinker = not self.driver.right_blinker
      self.driver.left_blinker = False
    elif cmd == "ignition":
      self.driver.ignition = not self.driver.ignition
    elif cmd == "restart":
      self.world.reset()
      self.past_startup_engaged = False
    elif cmd == "recenter":
      self.world.recenter()

  def run(self):
    # CAN and cameras go out on their own clocks, so they keep flowing while a frame renders
    threads = [threading.Thread(target=self._send_can, daemon=True), threading.Thread(target=self._send_cameras, daemon=True)]
    for t in threads:
      t.start()
    try:
      self._loop()
    finally:
      self.keep_alive = False
      self.world.close()
      for t in threads:
        t.join(timeout=1)

  def _send_can(self):
    rk = Ratekeeper(100, None)
    vehicle = self.world.vehicle
    while self.keep_alive:
      self.car.update(self.driver, vehicle.v, vehicle.steer.x, self.engaged)
      rk.keep_time()

  def _send_cameras(self):
    # like camerad, frames go out on a steady 20 Hz clock, with whatever was rendered last. modeld sees any frame it
    # misses as a dropped frame and invalidates its outputs, so frames must never bunch up after the sim hiccups
    next_frame_time = time.monotonic()
    last_images = None
    while self.keep_alive:
      next_frame_time = max(next_frame_time + DT_MDL, time.monotonic() + DT_MDL * MIN_FRAME_SPACING)
      time.sleep(max(0., next_frame_time - time.monotonic()))
      if (images := self.world.images) is not None:
        self.sensors.camerad.cam_send_yuv_road(images[0])
        self.sensors.camerad.cam_send_yuv_wide_road(images[1])
        self.repeated.append(images is last_images)
        last_images = images

  def _loop(self):
    driver, world, vehicle = self.driver, self.world, self.world.vehicle
    ended = None
    frame = 0
    next_frame_time = time.monotonic()
    behind = 0.  # how far the sim fell behind real time since the last status
    while self.keep_alive:
      while not self.cmd_q.empty():
        self.handle(self.cmd_q.get())
      now = time.monotonic()
      if now > self.manual_until:
        self.manual = dict.fromkeys(self.manual, 0.)

      driver.cruise_button = self.cruise_button if now < self.cruise_button_until else 0
      driver.gas = self.manual["gas"]
      driver.brake = self.manual["brake"]
      driver.steer_torque = self.manual["steer"] * -10000

      self.car.sm.update(0)
      selfdrive = self.car.sm['selfdriveState']
      self.engaged = engaged = selfdrive.active
      world.set_engaged(engaged)

      # the driver can always push the gas and turn the wheel, even while openpilot drives
      driver_steer = self.manual["steer"] * -40
      if engaged:
        actuators = self.car.sm['carControl'].actuators
        throttle = max(np.clip(actuators.accel / 1.6, 0.0, 1.0), driver.gas)
        brake = np.clip(-actuators.accel / 4.0, 0.0, 1.0)
        steer = actuators.steeringAngleDeg + driver_steer
        self.past_startup_engaged = True
        if world.ended is not None:
          driver.cruise_button = CruiseButtons.CANCEL
      else:
        throttle, brake, steer = driver.gas, driver.brake, driver_steer
        if not self.past_startup_engaged and selfdrive.engageable:
          driver.cruise_button = CruiseButtons.DECEL_SET if self.startup_button_prev else CruiseButtons.MAIN  # force engagement on startup
          self.startup_button_prev = not self.startup_button_prev

      world.apply_controls(steer, throttle, brake)
      self.sensors.update(vehicle)

      if frame % 5 == 0:
        world.tick()
        # catching up, only the latest frame renders. sending the ones in between back-to-back would show up as dropped frames to modeld
        if time.monotonic() - next_frame_time < DT_MDL:
          world.render()

      if world.ended != ended:
        ended = world.ended
        if ended is not None:
          self.status_q.put(("episode_end", world.status() | {"behind": behind, "repeated": sum(self.repeated)}))
      if frame % (100 // STATUS_HZ) == 0:
        self.status_q.put(("status", world.status() | {"behind": behind, "repeated": sum(self.repeated)}))
        behind = 0.

      # sim time keeps up with real time: a slow frame (a render, a busy machine) is made up with quicker ones after it,
      # instead of being dropped, so the cameras stay at 20 Hz. only a long stall resyncs
      frame += 1
      next_frame_time += DT_CTRL
      delay = next_frame_time - time.monotonic()
      behind = max(behind, -delay)
      if delay > 0:
        time.sleep(delay)
      elif delay < -MAX_CATCH_UP:
        next_frame_time = time.monotonic()
