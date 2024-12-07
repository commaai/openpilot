import ctypes
import multiprocessing
import socket
import pickle
import threading
import math
import csv
import time
from time import sleep
from multiprocessing import Pipe, Array
from openpilot.tools.sim.lib.common import SimulatorState, World, GPSState
from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
from openpilot.tools.sim.bridge.gz.msgs import Report, RequestType, Request
from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H
from openpilot.tools.sim.bridge.gz.msgs import GPSPos
from PIL import Image
from numpy import asarray
import numpy as np

class GZWorld(World):
  def __init__(self, status_q):
    super().__init__(False)
    self.status_q = status_q

    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "starting"))

    self.wide_camera_array = None
    # Dummy Camera from a file
    self.camera_array = Array(ctypes.c_uint8, W*H*3)
    self.road_image = np.frombuffer(self.camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    self.controls_send, self.controls_recv = Pipe()
    self.simulation_state_send, self.simulation_state_recv = Pipe()
    self.vehicle_state_send, self.vehicle_state_recv = Pipe()

    self.exit_event = multiprocessing.Event()
    self.op_engaged = multiprocessing.Event()

    self.test_run = False

    self.first_engage = None
    self.last_check_timestamp = 0
    self.distance_moved = 0

    self.vehicle_last_pos = [0,0,0]
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "started"))
    self.thread = threading.Thread(target=self.poll)
    self.thread.start()

    self.steer_ratio = 15
    self.velocity = vec3(0,0,0)
    self.bearing = 0
    self.steering = 0
    self.gps = GPSPos(0,0,0)
    self.vc = [0.0,0.0]
    self.reset_time = 0
    self.should_reset = False
    self.image_lock.release()


    self.file = open("/tmp/gz_log.csv", "w")
    self.writer = csv.writer(self.file)
    self.writer.writerow([
      "time",
      "velocity_x",
      "velocity_y",
      "velocity_z",
      "bearing",
      "steering_angle",
      "longitude",
      "latitude",
      "altitude"
      ])

  def poll(self):
    while not self.exit_event.is_set():
      client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      client.connect(("host.docker.internal", 8069))
      serialized = pickle.dumps(Request(RequestType.GET_REPORT))
      client.send(len(serialized).to_bytes(4, "little"))
      client.sendall(serialized)
      inlen = client.recv(4)
      inlen = int.from_bytes(inlen, "little")
      data = bytes()
      while len(data) < inlen:
        data += client.recv(inlen - len(data))
      msg = pickle.loads(data)
      client.close()
      if(len(msg.image.data) > 0):
        original= np.frombuffer(bytes(msg.image.data), dtype=np.uint8).reshape((msg.image.height, msg.image.width, 3))
        self.road_image[...] = np.pad(original, [(0, H - msg.image.height), (0, W - msg.image.width), (0,0)], mode='constant', constant_values=0)
      self.image_lock.release()
      self.velocity = vec3(float(msg.odometry.velocity[0]),
                           float(msg.odometry.velocity[1]),
                           0.0)
      self.bearing = msg.odometry.yaw
      self.gps = msg.gps
      self.steering = msg.odometry.heading

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    print(f"steer_angle: {steer_angle}, throttle_out: {throttle_out}, brake_out: {brake_out}")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("host.docker.internal", 8069))
    serialized = pickle.dumps(Request(RequestType.IN_COMMAND, [steer_angle, throttle_out, brake_out]))
    client.send(len(serialized).to_bytes(4, "little"))
    client.sendall(serialized)
    ilen = client.recv(4)
    ilen = int.from_bytes(ilen, "little")
    data = client.recv(ilen)
    if data != b'OK':
      print(f'Unexpected response: {data}')
    client.close()

  def read_cameras(self):
    pass

  def read_sensors(self, state: SimulatorState):
    state.velocity = self.velocity
    state.bearing = math.degrees(self.bearing)
    state.steering_angle = math.degrees(self.steering)
    state.gps = GPSState()
    state.gps.latitude = self.gps.latitude
    state.gps.longitude = self.gps.longitude
    state.gps.altitude = 0
    state.valid = True
    self.writer.writerow([
      time.monotonic(),
      state.velocity[0],
      state.velocity[1],
      state.velocity[2],
      state.bearing,
      state.steering_angle,
      state.gps.longitude,
      state.gps.latitude,
      state.gps.altitude
      ])


  def read_state(self):
    pass

  def tick(self):
    pass

  def reset(self):
    pass

  def close(self, reason: str):
    self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, reason))
