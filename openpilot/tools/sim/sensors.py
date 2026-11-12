import math
import time

from msgq.visionipc import VisionIpcServer
from openpilot.cereal import log
from openpilot.cereal.visionipc import VisionStreamType
import openpilot.cereal.messaging as messaging

from openpilot.common.realtime import DT_DMON
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.sim.vehicle import Vehicle

W, H = 1928, 1208
# the car drives on a plane, laid over the globe here
BASE_LAT, BASE_LON = 32.75308505188913, -117.2095393365393
DEG_TO_METERS = 100000


class Camerad:
  """Simulates the camerad daemon"""
  def __init__(self):
    self.pm = messaging.PubMaster(['narrowRoadCameraState', 'wideRoadCameraState'])

    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    stride, y_height, _, size = get_nv12_info(W, H)
    buffer_args = (5, W, H, size, stride, stride * y_height)

    self.vipc_server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_NARROW_ROAD, *buffer_args)
    self.vipc_server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_WIDE_ROAD, *buffer_args)

    self.vipc_server.start_listener()

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'narrowRoadCameraState', VisionStreamType.VISION_STREAM_NARROW_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    dat = messaging.new_message(pub_type, valid=True)
    self.vipc_server.send(yuv_type, yuv, frame_id, dat.logMonoTime, dat.logMonoTime)

    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)


class SimulatedSensors:
  """Simulates the comma device's sensors (IMU, GPS, peripherals, driver monitoring) and camerad"""

  def __init__(self):
    self.pm = messaging.PubMaster(['accelerometer', 'gyroscope', 'gpsLocationExternal', 'driverStateV2', 'driverMonitoringState', 'peripheralState'])
    self.camerad = Camerad()
    self.frame = 0
    self.last_perp_update = 0.
    self.last_dmon_update = 0.

  def send_imu_message(self, vehicle: Vehicle):
    # IMU sensor axes are (-z, -y, -x) of the device frame
    dat = messaging.new_message('accelerometer', valid=True)
    dat.accelerometer.timestamp = dat.logMonoTime
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = [9.81, -vehicle.v * vehicle.yaw_rate, -vehicle.accel.x]
    self.pm.send('accelerometer', dat)

    dat = messaging.new_message('gyroscope', valid=True)
    dat.gyroscope.timestamp = dat.logMonoTime
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = [-vehicle.yaw_rate, 0., 0.]
    self.pm.send('gyroscope', dat)

  def send_gps_message(self, vehicle: Vehicle):
    x, y, heading = map(float, vehicle.pose)
    dat = messaging.new_message('gpsLocationExternal', valid=True)
    dat.gpsLocationExternal = {
      "unixTimestampMillis": int(time.time() * 1000),  # noqa: TID251
      "flags": 1,  # valid fix
      "horizontalAccuracy": 1.0,
      "verticalAccuracy": 1.0,
      "speedAccuracy": 0.1,
      "bearingAccuracyDeg": 0.1,
      "vNED": [-vehicle.v * math.sin(heading), vehicle.v * math.cos(heading), 0.],
      "bearingDeg": math.degrees(heading),
      "latitude": BASE_LAT + x / DEG_TO_METERS,
      "longitude": BASE_LON + y / DEG_TO_METERS,
      "altitude": 0.,
      "speed": vehicle.v,
      "source": log.GpsLocationData.SensorSource.ublox,
    }
    self.pm.send('gpsLocationExternal', dat)

  def send_peripheral_state(self):
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    self.pm.send('peripheralState', dat)

  def send_fake_driver_monitoring(self):
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    dat.driverStateV2.rightDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.rightDriverData.faceProb = 1.0
    self.pm.send('driverStateV2', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState', valid=True)
    dm = dat.driverMonitoringState
    dm.alertLevel = log.DriverMonitoringState.AlertLevel.none
    dm.activePolicy = log.DriverMonitoringState.MonitoringPolicy.vision
    dm.visionPolicyState.faceDetected = True
    dm.visionPolicyState.isDistracted = False
    dm.visionPolicyState.awarenessPercent = 100
    self.pm.send('driverMonitoringState', dat)

  def update(self, vehicle: Vehicle):
    """Called at 100 Hz"""
    now = time.monotonic()
    self.send_imu_message(vehicle)
    if self.frame % 10 == 0:
      self.send_gps_message(vehicle)
    self.frame += 1

    if (now - self.last_dmon_update) > DT_DMON/2:
      self.send_fake_driver_monitoring()
      self.last_dmon_update = now

    if (now - self.last_perp_update) > 0.25:
      self.send_peripheral_state()
      self.last_perp_update = now
