import threading
import numpy as np
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.basedir import BASEDIR
from common.realtime import DT_DMON, Ratekeeper
import cereal.messaging as messaging
import pyopencl as cl
import pyopencl.array as cl_array
import os
import time
from cereal import log
from tools.streamer.can import simulator_can_function
from tools.streamer.includes import VehicleState
from common.params import Params

SCALE = 1
W, H = 1928 // SCALE, 1208 // SCALE
REPEAT_COUNTER = 5
PRINT_DECIMATION = 100
STEER_RATIO = 15.


class Tasks:
  def __init__(self, W, H):
    self._threads = []
    self._exit_event = threading.Event()
    self._exit_event.clear()
    self.vs = VehicleState()
    self.W = W
    self.H = H

  def panda(self):
    panda_state_function(self.vs, self._exit_event)

  def peripherals(self):
    peripheral_state_function(self._exit_event)
  
  def dm(self):
    fake_driver_monitoring(self._exit_event)

  def can(self):
    can_function_runner(self.vs, self._exit_event)

  def sensors(self):
    sensor_function(self.vs, self._exit_event)

  def device_state(self):
    device_state_function(self._exit_event)

class Camerad:
  def __init__(self, W, H):
    self.pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState'])
    self.wide_W = W
    self.wide_H = H
    self.road_W = W # possibly send a different resolution for the road camera since we just scale it up to match the wide camera
    self.road_H = H
    print("wide resolution: ", self.wide_W, self.wide_H)
    print("road resolution: ", self.road_W, self.road_H)
    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.wide_vipc_type = VisionStreamType.VISION_STREAM_WIDE_ROAD
    self.road_vipc_type = VisionStreamType.VISION_STREAM_ROAD
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(self.road_vipc_type, 5, False, self.road_W, self.road_H)
    self.vipc_server.create_buffers(self.wide_vipc_type, 5, False, self.wide_W, self.wide_H)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)

    # road
    road_cl_arg = f" -DHEIGHT={self.road_H} -DWIDTH={self.road_W} -DRGB_STRIDE={self.road_W * 3} -DUV_WIDTH={self.road_W // 2} -DUV_HEIGHT={self.road_H // 2} -DRGB_SIZE={self.road_W * self.road_H} -DCL_DEBUG "
    kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(road_cl_arg)
      self.road_krnl = prg.rgb_to_nv12
    self.road_Wdiv4 = self.road_W // 4 if (self.road_W % 4 == 0) else (self.road_W + (4 - self.road_W % 4)) // 4
    self.road_Hdiv4 = self.road_H // 4 if (self.road_H % 4 == 0) else (self.road_H + (4 - self.road_H % 4)) // 4

    # wide
    wide_cl_arg = f" -DHEIGHT={self.wide_H} -DWIDTH={self.wide_W} -DRGB_STRIDE={self.wide_W * 3} -DUV_WIDTH={self.wide_W // 2} -DUV_HEIGHT={self.wide_H // 2} -DRGB_SIZE={self.wide_W * self.wide_H} -DCL_DEBUG "
    kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12_opy.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(wide_cl_arg)
      self.wide_krnl = prg.rgb_to_nv12_opy
    self.wide_Wdiv4 = self.wide_W // 4 if (self.wide_W % 4 == 0) else (self.wide_W + (4 - self.wide_W % 4)) // 4
    self.wide_Hdiv4 = self.wide_H // 4 if (self.wide_H % 4 == 0) else (self.wide_H + (4 - self.wide_H % 4)) // 4

  def cam_callback_road(self, image):
    rgb = image.copy()

    rgb_cl = cl_array.to_device(self.queue, rgb) # RGB cl
    yuv_cl = cl_array.empty_like(rgb_cl) # YUV cl
    self.road_krnl(self.queue, (np.int32(self.road_Wdiv4), np.int32(self.road_Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait() # YUV cl (async)
    
    yuv = np.resize(yuv_cl.get(), rgb.size // 2) # YUV
    eof = int(self.frame_road_id * 0.5 * 1e9) # 20 fps
    
    self.vipc_server.send(self.road_vipc_type, yuv.data.tobytes(), self.frame_road_id, eof, eof) # YUV

    dat = messaging.new_message('roadCameraState')
    msg = {
      "frameId": self.frame_road_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat,'roadCameraState', msg)
    self.pm.send('roadCameraState', dat)

    self.frame_road_id += 1

  def cam_callback_wide_road(self, image):
    rgb = image.copy()

    rgb_cl = cl_array.to_device(self.queue, rgb) # RGB cl
    yuv_cl = cl_array.empty_like(rgb_cl) # YUV cl
    self.wide_krnl(self.queue, (np.int32(self.wide_Wdiv4), np.int32(self.wide_Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait() # YUV cl (async)
    
    yuv = np.resize(yuv_cl.get(), rgb.size // 2) # YUV
    eof = int(self.frame_wide_id * 0.5 * 1e9) # 20 fps
    
    self.vipc_server.send(self.wide_vipc_type, yuv.data.tobytes(), self.frame_wide_id, eof, eof) # YUV

    dat = messaging.new_message('wideRoadCameraState')
    msg = {
      "frameId": self.frame_road_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, 'wideRoadCameraState', msg)
    self.pm.send('wideRoadCameraState', dat)

    self.frame_wide_id += 1

    

def panda_state_function(vs: VehicleState, exit_event: threading.Event):
  pm = messaging.PubMaster(['pandaStates'])
  while not exit_event.is_set():
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': vs.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': vs.is_engaged,
      'safetyModel': 'simulator'
    }
    pm.send('pandaStates', dat)
    time.sleep(0.5)

def device_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['deviceState'])
  #rk = Ratekeeper(20, print_delay_threshold=None)
  while not exit_event.is_set():
    dat = messaging.new_message('deviceState')
    dat.deviceState.freeSpacePercent = 100
    dat.deviceState.memoryUsagePercent = 0
    pm.send('deviceState', dat)

def peripheral_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['peripheralState'])
  while not exit_event.is_set():
    dat = messaging.new_message('peripheralState')
    Params().put_bool("ObdMultiplexingDisabled", True)
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

def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverStateV2', 'driverMonitoringState'])
  while not exit_event.is_set():
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    dat.driverStateV2.rightDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.rightDriverData.faceProb = 1.0
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
  rk = Ratekeeper(100, print_delay_threshold=.05)
  pm = messaging.PubMaster(['can'])
  while not exit_event.is_set():
    simulator_can_function(pm, vs.speed, vs.angle, i, vs.cruise_sp, vs.is_engaged)
    rk.keep_time()
    i += 1

def sensor_function(vehicle_state, exit_event: threading.Event):
  pm = messaging.PubMaster(["gpsLocationExternal",'accelerometer','gyroscope'])
  Params().put_bool("UbloxAvailable", False)
  
  
  # transform vel from carla to NED
  # north is -Y in CARLA

  while not exit_event.is_set():
    dat = messaging.new_message('gpsLocationExternal')
    velNED = [
      -0,  # north/south component of NED is negative when moving south
      0,  # positive when moving east, which is x in carla
      0,
    ]
    dat.gpsLocationExternal = {
      "unixTimestampMillis": int(time.time() * 1000),
      "flags": 1,  # valid fix
      "accuracy": 1.0,
      "verticalAccuracy": 1.0,
      "speedAccuracy": 0.1,
      "bearingAccuracyDeg": 0.1,
      "vNED": velNED,
      "bearingDeg": 0,
      "latitude": 0,
      "longitude": 0,
      "altitude": 0,
      "speed": 0,
      "source": 0,
    }
    pm.send('gpsLocationExternal', dat)

    vehicle_state.bearing_deg = 0
    dat = messaging.new_message('accelerometer')
    dat.accelerometer.sensor = 4
    dat.accelerometer.type = 0x10
    dat.accelerometer.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = [0, 0, 0]
    pm.send('accelerometer', dat)

    # copied these numbers from locationd
    dat = messaging.new_message('gyroscope')
    dat.gyroscope.sensor = 5
    dat.gyroscope.type = 0x10
    dat.gyroscope.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = [0, 0, 0]
    
    pm.send('gyroscope', dat)
   
    
    time.sleep(0.01)

def streamer():
  import zmq
  import struct
  import cv2
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.setsockopt(zmq.CONFLATE, 1)
  socket.set_hwm(5)
  socket.connect("ipc:///tmp/metadrive_window")
  c_W = 1928
  c_H = 1208
  _camerad = Camerad(c_W, c_H)
  while True:
      
    image_buffer_msg = socket.recv()
    W, H = struct.unpack('ii', image_buffer_msg[:8]) # read the first 32bit int as W, the second as H to handle different resolutions
    image_buffer = image_buffer_msg[8:]
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape((H, W, 4))
    image = image[:, :, :3] # remove the alpha channel
    # flip the image with numpy
    image = np.flip(image, 0)
    #crop the image. idk why but 60deg crop messes up the model? use 5 instead?
    cropped_img = image[H//5:H-(H//5), W//5:W-(W//5)]
    road_img = cv2.resize(cropped_img, (c_W, c_H))
    #scale the image to 1928*1208
    wide_img = cv2.resize(image, (c_W, c_H))
    _camerad.cam_callback_wide_road(wide_img)
    _camerad.cam_callback_road(road_img)

        