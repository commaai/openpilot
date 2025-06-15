from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class SensorReading:
  measurement: np.ndarray
  timestamp: float
  valid: bool = True


class ISensorProvider(ABC):
  @abstractmethod
  def get_accelerometer_reading(self, msg_data: dict, t: float) -> SensorReading | None:
    pass

  @abstractmethod
  def get_gyroscope_reading(self, msg_data: dict, t: float) -> SensorReading | None:
    pass

  @abstractmethod
  def get_critical_services(self) -> list[str]:
    pass

  @abstractmethod
  def requires_hardware_sensors(self) -> bool:
    pass


class HardwareSensorProvider(ISensorProvider):
  """Provides sensor data from dedicated IMU hardware"""

  def get_accelerometer_reading(self, msg_data: dict, t: float) -> SensorReading | None:
    if 'accelerometer' not in msg_data:
      return None
    msg = msg_data['accelerometer']
    sensor_time = msg.timestamp * 1e-9
    v = msg.acceleration.v
    # Apply the same transformation as original code: [-v[2], -v[1], -v[0]]
    meas = np.array([-v[2], -v[1], -v[0]])
    return SensorReading(meas, sensor_time, msg.valid)

  def get_gyroscope_reading(self, msg_data: dict, t: float) -> SensorReading | None:
    if 'gyroscope' not in msg_data:
      return None
    msg = msg_data['gyroscope']
    sensor_time = msg.timestamp * 1e-9
    v = msg.gyroUncalibrated.v
    # Apply the same transformation as original code: [-v[2], -v[1], -v[0]]
    meas = np.array([-v[2], -v[1], -v[0]])
    return SensorReading(meas, sensor_time, msg.valid)

  def get_critical_services(self) -> list[str]:
    return ["accelerometer", "gyroscope", "cameraOdometry"]

  def requires_hardware_sensors(self) -> bool:
    return True


class CarStateSensorProvider(ISensorProvider):
  """Provides fallback sensor data from vehicle CAN bus"""

  def get_accelerometer_reading(self, msg_data: dict, t: float) -> SensorReading | None:
    if 'carState' not in msg_data:
      return None
    msg = msg_data['carState']
    # Longitudinal acceleration + gravity vector
    meas = np.array([msg.aEgo, 0, -9.81])
    return SensorReading(meas, t, msg.valid)

  def get_gyroscope_reading(self, msg_data: dict, t: float) -> SensorReading | None:
    if 'carState' not in msg_data:
      return None
    msg = msg_data['carState']
    # Only yaw rate available from vehicle
    meas = np.array([0, 0, -msg.yawRate])
    return SensorReading(meas, t, msg.valid)

  def get_critical_services(self) -> list[str]:
    return ["cameraOdometry"]  # Only camera needed

  def requires_hardware_sensors(self) -> bool:
    return False


def get_sensor_provider() -> ISensorProvider:
  """Factory function to get appropriate sensor provider"""
  from openpilot.system.hardware import PC

  return CarStateSensorProvider() if PC else HardwareSensorProvider()
