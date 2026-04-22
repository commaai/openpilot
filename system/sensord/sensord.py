#!/usr/bin/env python3
import os
import time
import math
import ctypes
import select

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.gpio import gpioevent_data

from openpilot.system.sensord.sensors.i2c_sensor import Sensor
from openpilot.system.sensord.sensors.lsm6ds3_accel import LSM6DS3_Accel
from openpilot.system.sensord.sensors.lsm6ds3_gyro import LSM6DS3_Gyro
from openpilot.system.sensord.sensors.lsm6ds3_temp import LSM6DS3_Temp

I2C_BUS_IMU = 1

def publish_sensor(pm: messaging.PubMaster, sensor: Sensor, ts: int | None = None) -> None:
  evt = sensor.get_event(ts)
  if not sensor.is_data_valid():
    return

  msg = messaging.new_message(sensor.service, valid=True)
  setattr(msg, sensor.service, evt)
  pm.send(sensor.service, msg)

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

  initialized_sensors: list[Sensor] = []
  irq_sensors: dict[int, Sensor] = {}
  polled_sensors: list[Sensor] = []
  for sensor in sensors:
    try:
      sensor.init()
      initialized_sensors.append(sensor)
      if sensor.irq_pin is None:
        polled_sensors.append(sensor)
      else:
        irq_sensors[sensor.open_irq_fd()] = sensor
    except Exception:
      cloudlog.exception(f"Error initializing {sensor.service} sensor")

  pm = messaging.PubMaster([sensor.service for sensor in initialized_sensors])
  poller = select.poll()
  for fd in irq_sensors:
    poller.register(fd, select.POLLIN | select.POLLPRI)

  poll_periods = {
    sensor: int(1e9 / SERVICE_LIST[sensor.service].frequency)
    for sensor in polled_sensors
  }
  next_poll = {
    sensor: time.monotonic_ns()
    for sensor in polled_sensors
  }
  event_size = ctypes.sizeof(gpioevent_data)
  offset = time.time_ns() - time.monotonic_ns()

  try:
    while True:
      timeout_ms = -1
      if len(next_poll):
        timeout_ns = min(next_poll.values()) - time.monotonic_ns()
        timeout_ms = max(0, math.ceil(timeout_ns / 1e6))

      for fd, event in poller.poll(timeout_ms):
        if not (event & (select.POLLIN | select.POLLPRI)):
          continue

        sensor = irq_sensors[fd]
        dat = os.read(fd, event_size * 16)
        cur_offset = time.time_ns() - time.monotonic_ns()
        if abs(cur_offset - offset) > 10 * 1e6:  # ms
          cloudlog.warning(f"{sensor.service} time jumped: {cur_offset} {offset}")
          offset = cur_offset
          continue

        for i in range(0, len(dat), event_size):
          evd = gpioevent_data.from_buffer_copy(dat[i:i+event_size])
          try:
            publish_sensor(pm, sensor, evd.timestamp - cur_offset)
          except Sensor.DataNotReady:
            pass
          except Exception:
            cloudlog.exception(f"Error processing {sensor.service}")

      now = time.monotonic_ns()
      for sensor in polled_sensors:
        if now < next_poll[sensor]:
          continue

        try:
          publish_sensor(pm, sensor)
        except Exception:
          cloudlog.exception(f"Error in {sensor.service} polling loop")
        next_poll[sensor] += poll_periods[sensor]
  except KeyboardInterrupt:
    pass
  finally:
    for sensor in initialized_sensors:
      try:
        sensor.shutdown()
      except Exception:
        cloudlog.exception("Error shutting down sensor")
      finally:
        sensor.close()

if __name__ == "__main__":
  main()
