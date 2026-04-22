#!/usr/bin/env python3
import time
import threading

from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog

from openpilot.system.sensord.sensors.i2c_sensor import Sensor
from openpilot.system.sensord.sensors.lsm6ds3_accel import LSM6DS3_Accel
from openpilot.system.sensord.sensors.lsm6ds3_gyro import LSM6DS3_Gyro
from openpilot.system.sensord.sensors.lsm6ds3_temp import LSM6DS3_Temp

I2C_BUS_IMU = 1

def main() -> None:
  config_realtime_process([1, ], 1)

  sensors: list[Sensor] = [
    LSM6DS3_Accel(I2C_BUS_IMU),
    LSM6DS3_Gyro(I2C_BUS_IMU),
    LSM6DS3_Temp(I2C_BUS_IMU),
  ]

  # Reset sensors
  for sensor in sensors:
    try:
      sensor.reset()
    except Exception:
      cloudlog.exception(f"Error initializing {sensor} sensor")

  exit_event = threading.Event()
  threads: list[threading.Thread] = []
  initialized_sensors: list[Sensor] = []
  for sensor in sensors:
    try:
      sensor.init()
      initialized_sensors.append(sensor)
      threads.append(threading.Thread(target=sensor.run, args=(exit_event,), daemon=True, name=sensor.service))
    except Exception:
      cloudlog.exception(f"Error initializing {sensor.service} sensor")

  try:
    for t in threads:
      t.start()
    while any(t.is_alive() for t in threads):
      time.sleep(1)
  except KeyboardInterrupt:
    pass
  finally:
    exit_event.set()
    for t in threads:
      if t.is_alive():
        t.join()

    for sensor in initialized_sensors:
      try:
        sensor.shutdown()
      except Exception:
        cloudlog.exception("Error shutting down sensor")
      finally:
        sensor.close()

if __name__ == "__main__":
  main()
