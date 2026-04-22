#!/usr/bin/env python3
import os
import time
import ctypes
import select
import threading

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.utils import sudo_write
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.gpio import get_irqs_for_action, gpiochip_get_ro_value_fd, gpioevent_data

from openpilot.system.hardware.tici.pins import GPIO
from openpilot.system.sensord.sensors.i2c_sensor import Sensor
from openpilot.system.sensord.sensors.lsm6ds3_accel import LSM6DS3_Accel
from openpilot.system.sensord.sensors.lsm6ds3_gyro import LSM6DS3_Gyro
from openpilot.system.sensord.sensors.lsm6ds3_temp import LSM6DS3_Temp

I2C_BUS_IMU = 1

def interrupt_loop(sensors: list[tuple[Sensor, str, bool]], event) -> None:
  interrupt_fds = {
    gpiochip_get_ro_value_fd("sensord", 0, GPIO.LSM_INT if service == "accelerometer" else GPIO.LSM_INT2): (sensor, service)
    for sensor, service, interrupt in sensors if interrupt
  }
  pm = messaging.PubMaster([service for _, service in interrupt_fds.values()])

  # Requesting both edges as the data ready pulse from the lsm6ds sensor is
  # very short (75us) and is mostly detected as falling edge instead of rising.
  # So if it is detected as rising the following falling edge is skipped.
  for irq in get_irqs_for_action("sensord"):
    sudo_write('1\n', f"/proc/irq/{irq}/smp_affinity_list")

  offset = time.time_ns() - time.monotonic_ns()
  event_size = ctypes.sizeof(gpioevent_data)

  poller = select.poll()
  for fd in interrupt_fds:
    poller.register(fd, select.POLLIN | select.POLLPRI)
  while not event.is_set():
    events = poller.poll(100)
    if not events:
      cloudlog.error("poll timed out")
      continue

    cur_offset = time.time_ns() - time.monotonic_ns()
    if abs(cur_offset - offset) > 10 * 1e6:  # ms
      cloudlog.warning(f"time jumped: {cur_offset} {offset}")
      offset = cur_offset
      continue

    for fd, flags in events:
      if not (flags & (select.POLLIN | select.POLLPRI)):
        cloudlog.error("no poll events set")
        continue

      sensor, service = interrupt_fds[fd]
      dat = os.read(fd, event_size * 16)
      for i in range(0, len(dat), event_size):
        evd = gpioevent_data.from_buffer_copy(dat[i:i+event_size])
        try:
          evt = sensor.get_event(evd.timestamp - cur_offset)
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
