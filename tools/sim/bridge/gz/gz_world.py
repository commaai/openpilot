import capnp
capnp.remove_import_hook()
import ctypes
import multiprocessing
import socket
import pickle
import threading
import math
import csv
import time
import json
from time import sleep
from multiprocessing import Pipe, Array
from openpilot.tools.sim.lib.common import SimulatorState, World, GPSState
from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H
from PIL import Image
from numpy import asarray
import numpy as np
MAX_BUFFER_SIZE = 10*1024*1024  # 10MB
log_capnp = capnp.load("/workspaces/openpilot/cereal/log.capnp")
car_capnp = capnp.load("/workspaces/openpilot/cereal/car.capnp")

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
    self.gps = GPSState()
    self.vc = [0.0,0.0]
    self.reset_time = 0
    self.should_reset = False
    self.image_lock.release()


    self.sensors_file = open("/tmp/sensors_log.csv", "w")
    self.sensors_log = csv.writer(self.sensors_file)
    self.sensors_log.writerow([
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
    self.control_file = open("/tmp/control_log.csv", "w")
    self.control_log = csv.writer(self.control_file)
    self.control_log.writerow([
      "time",
      "throttle",
      "brake",
      "steer"
    ])

  def poll(self):
    while not self.exit_event.is_set():
        data = bytearray(MAX_BUFFER_SIZE)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("host.docker.internal", 4069))
            s.send(bytes([0]))
            offset = 0
            while offset < len(data):
                recv = s.recv(len(data) - offset)
                data[offset:offset+len(recv)] = recv
                offset += len(recv)
                if len(recv) == 0:
                    break

        thumbnail = log_capnp.Thumbnail.from_bytes_packed(data)
        if(len(thumbnail.thumbnail) > 0):
            original= np.frombuffer(thumbnail.thumbnail, dtype=np.uint8).reshape((1080, 1920, 3))
            self.road_image[...] = np.pad(original, [(0, H - 1080), (0, W - 1920), (0,0)], mode='constant', constant_values=0)
        self.image_lock.release()

        data = bytearray(MAX_BUFFER_SIZE)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("host.docker.internal", 4069))
            s.send(bytes([1]))
            offset = 0
            while offset < len(data):
                recv = s.recv(len(data) - offset)
                data[offset:offset+len(recv)] = recv
                offset += len(recv)
                if len(recv) == 0:
                    break

        gpsData = log_capnp.GpsLocationData.from_bytes_packed(data)
        self.velocity = gpsData.vNED
        self.gps.latitude = gpsData.latitude
        self.gps.longitude = gpsData.longitude
        self.gps.altitude = gpsData.altitude
        self.bearing = gpsData.bearingDeg
        self.steering = gpsData.speed

  def start_vehicle(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("host.docker.internal", 4069))
            s.send(bytes([3]))
            confirmed = s.recv(1)
            if len(confirmed) == 0:
              raise Exception("Failed to start vehicle")

            if confirmed[0] != 0:
              raise Exception("Failed to start vehicle")


  def apply_controls(self, steer_angle, throttle_out, brake_out):
    self.control_log.writerow([time.time(), throttle_out, brake_out, steer_angle])
    actuators = car_capnp.CarControl.Actuators.new_message(
            steeringAngleDeg = steer_angle,
            gas = throttle_out)
    actuators.brake = brake_out
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("host.docker.internal", 4069))
        s.send(bytes([2]))
        s.sendall(actuators.to_bytes_packed())

  def read_cameras(self):
    pass

  def read_sensors(self, state: SimulatorState):
    state.velocity = vec3(
        self.velocity[0],
        self.velocity[1],
        0
        )
    state.bearing = self.bearing
    state.steering_angle = self.steering
    state.gps = GPSState()
    state.gps.latitude = self.gps.latitude
    state.gps.longitude = self.gps.longitude
    state.gps.altitude = 0
    state.valid = True
    self.sensors_log.writerow([
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
