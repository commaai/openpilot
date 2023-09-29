import ctypes
import functools
import multiprocessing
import numpy as np
import time

from multiprocessing import Pipe, Array
from openpilot.tools.sim.bridge.metadrive.metadrive_process import metadrive_process, metadrive_state
from openpilot.tools.sim.lib.common import SimulatorState, World
from openpilot.tools.sim.lib.camerad import W, H


class MetaDriveWorld(World):
  def __init__(self, config, dual_camera = False):
    super().__init__(dual_camera)
    self.camera_array = Array(ctypes.c_uint8, W*H*3)
    self.road_image = np.frombuffer(self.camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    self.controls_send, self.controls_recv = Pipe()
    self.state_send, self.state_recv = Pipe()

    self.exit_event = multiprocessing.Event()

    self.metadrive_process = multiprocessing.Process(name="metadrive process", target=
                              functools.partial(metadrive_process, dual_camera, config,
                                                self.camera_array, self.controls_recv, self.state_send, self.exit_event))
    self.metadrive_process.start()

    print("----------------------------------------------------------")
    print("---- Spawning Metadrive world, this might take awhile ----")
    print("----------------------------------------------------------")

    self.state_recv.recv() # wait for a state message to ensure metadrive is launched

    self.steer_ratio = 15
    self.vc = [0.0,0.0]
    self.reset_time = 0
    self.should_reset = False

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    if (time.monotonic() - self.reset_time) > 5:
      self.vc[0] = steer_angle

      if throttle_out:
        self.vc[1] = throttle_out/10
      else:
        self.vc[1] = -brake_out
    else:
      self.vc[0] = 0
      self.vc[1] = 0

    self.controls_send.send([*self.vc, self.should_reset])

  def read_sensors(self, state: SimulatorState):
    while self.state_recv.poll(0):
      md_state: metadrive_state = self.state_recv.recv()
      state.velocity = md_state.velocity
      state.bearing = md_state.bearing
      state.steering_angle = md_state.steering_angle
      state.gps.from_xy(md_state.position)
      state.valid = True

  def read_cameras(self):
    pass

  def tick(self):
    pass

  def reset(self):
    self.should_reset = True

  def close(self):
    self.exit_event.set()
    self.metadrive_process.join()
