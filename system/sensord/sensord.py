#!/usr/bin/env python3
import os
import time
import ctypes
import select
import threading

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.util import sudo_write
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.gpio import gpiochip_get_ro_value_fd, gpioevent_data

from openpilot.system.sensord.sensors.i2c_sensor import Sensor
from openpilot.system.sensord.sensors.lsm6ds3_accel import LSM6DS3_Accel
from openpilot.system.sensord.sensors.lsm6ds3_gyro import LSM6DS3_Gyro
from openpilot.system.sensord.sensors.lsm6ds3_temp import LSM6DS3_Temp
from openpilot.system.sensord.sensors.mmc5603nj_magn import MMC5603NJ_Magn

I2C_BUS_IMU = 1

def interrupt_loop(sensors: list[tuple[Sensor, str, bool]], event) -> None:
  pm = messaging.PubMaster([service for sensor, service, interrupt in sensors if interrupt])

  # Requesting both edges as the data ready pulse from the lsm6ds sensor is
  # very short (75us) and is mostly detected as falling edge instead of rising.
  # So if it is detected as rising the following falling edge is skipped.
  fd = gpiochip_get_ro_value_fd("sensord", 0, 84)

  # Configure IRQ affinity
  irq_path = "/proc/irq/336/smp_affinity_list"
  if not os.path.exists(irq_path):
    irq_path = "/proc/irq/335/smp_affinity_list"
  if os.path.exists(irq_path):
    sudo_write('1\n', irq_path)

  offset = time.time_ns() - time.monotonic_ns()

  poller = select.poll()
  poller.register(fd, select.POLLIN | select.POLLPRI)
  while not event.is_set():
    events = poller.poll(100)
    if not events:
      cloudlog.error("poll timed out")
      continue
    if not (events[0][1] & (select.POLLIN | select.POLLPRI)):
      cloudlog.error("no poll events set")
      continue

    dat = os.read(fd, ctypes.sizeof(gpioevent_data)*16)
    evd = gpioevent_data.from_buffer_copy(dat)

    cur_offset = time.time_ns() - time.monotonic_ns()
    if abs(cur_offset - offset) > 10 * 1e6:  # ms
      cloudlog.warning(f"time jumped: {cur_offset} {offset}")
      offset = cur_offset
      continue

    ts = evd.timestamp - cur_offset
    for sensor, service, interrupt in sensors:
      if interrupt:
        try:
          evt = sensor.get_event(ts)
          if not sensor.is_data_valid():
            continue
          msg = messaging.new_message(service, valid=True)
          setattr(msg, service, evt)
          pm.send(service, msg)
        except Sensor.DataNotReady:
          pass
        except Exception:
          cloudlog.exception(f"Error processing {service}")


def polling_loop(sensor: Sensor, service: str, event: threading.Event) -> None:
  pm = messaging.PubMaster([service])
  rk = Ratekeeper(SERVICE_LIST[service].frequency, print_delay_threshold=None)
  while not event.is_set():
    try:
      evt = sensor.get_event()
      if not sensor.is_data_valid():
        continue
      msg = messaging.new_message(service, valid=True)
      setattr(msg, service, evt)
      pm.send(service, msg)
    except Exception:
      cloudlog.exception(f"Error in {service} polling loop")
    rk.keep_time()

def main() -> None:
  config_realtime_process([1, ], 1)

  sensors_cfg = [
    (LSM6DS3_Accel(I2C_BUS_IMU), "accelerometer", True),
    (LSM6DS3_Gyro(I2C_BUS_IMU), "gyroscope", True),
    (LSM6DS3_Temp(I2C_BUS_IMU), "temperatureSensor", False),
    (MMC5603NJ_Magn(I2C_BUS_IMU), "magnetometer", False),
  ]

  # Reset sensors
  for sensor, _, _ in sensors_cfg:
    try:
      sensor.reset()
    except Exception:
      cloudlog.exception(f"Error initializing {sensor} sensor")

  # Initialize sensors
  exit_event = threading.Event()
  threads = [
    threading.Thread(target=interrupt_loop, args=(sensors_cfg, exit_event), daemon=True)
  ]
  for sensor, service, interrupt in sensors_cfg:
    try:
      sensor.init()
      if not interrupt:
        # Start polling thread for sensors without interrupts
        threads.append(threading.Thread(
          target=polling_loop,
          args=(sensor, service, exit_event),
          daemon=True
        ))
    except Exception:
      cloudlog.exception(f"Error initializing {service} sensor")

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

    for sensor, _, _ in sensors_cfg:
      try:
        sensor.shutdown()
      except Exception:
        cloudlog.exception("Error shutting down sensor")

if __name__ == "__main__":
  main()
