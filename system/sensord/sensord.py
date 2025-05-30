#!/usr/bin/env python3
import os
import time
import threading

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog

from openpilot.system.sensord.sensors.i2c_sensor import I2CSensor
from openpilot.system.sensord.sensors.lsm6ds3_accel import LSM6DS3_Accel
from openpilot.system.sensord.sensors.lsm6ds3_gyro import LSM6DS3_Gyro
from openpilot.system.sensord.sensors.lsm6ds3_temp import LSM6DS3_Temp
from openpilot.system.sensord.sensors.mmc5603nj_magn import MMC5603NJ_Magn

I2C_BUS_IMU = 1
GPIO_LSM_INT = 0  # Update this with actual GPIO number if needed

def interrupt_loop(sensors: list[tuple[I2CSensor, str]], event) -> None:
  pm = messaging.PubMaster([service for sensor, service in sensors if sensor.gpio_nr is not None])

  #gpio_fd: int | None = None
  #for sensor, _ in sensors:
  #  if sensor.gpio_nr is not None:
  #    gpio_fd = sensor.gpio_line.fd()
  #    break
  #assert gpio_fd is not None

  #poller = select.poll()
  #poller.register(gpio_fd, select.POLLIN | select.POLLPRI)
  while not event.is_set():
    #events = poller.poll(100)
    time.sleep(1./104)
    for sensor, service in sensors:
      if sensor.gpio_nr is None:
        continue

      try:
        msg = messaging.new_message(service)
        setattr(msg, service, sensor.get_event())
        pm.send(service, msg)
      except Exception:
        cloudlog.exception(f"Error processing {service}")

def polling_loop(sensor: I2CSensor, service: str, event: threading.Event) -> None:
  pm = messaging.PubMaster([service])
  rk = Ratekeeper(SERVICE_LIST[service].frequency, print_delay_threshold=None)
  while not event.is_set():
    try:
      sensor_event = sensor.get_event()
      if sensor_event is not None:
        msg = messaging.new_message(service)
        setattr(msg, service, sensor.get_event())
        pm.send(service, msg)
      rk.keep_time()
    except Exception:
      cloudlog.exception(f"Error in {service} polling loop")

def main() -> None:
  config_realtime_process([1, ], 1)

  exit_event = threading.Event()
  sensors_cfg = [
    (LSM6DS3_Accel(I2C_BUS_IMU, GPIO_LSM_INT), "accelerometer"),
    (LSM6DS3_Gyro(I2C_BUS_IMU, GPIO_LSM_INT, True), "gyroscope"),
    (LSM6DS3_Temp(I2C_BUS_IMU), "temperatureSensor"),
    (MMC5603NJ_Magn(I2C_BUS_IMU), "magnetometer"),
  ]

  # Initialize sensors
  threads = []
  for sensor, service in sensors_cfg:
    try:
      sensor.init()
      if sensor.gpio_nr is None:
        # Start polling thread for sensors without interrupts
        t = threading.Thread(
          target=polling_loop,
          args=(sensor, service, exit_event),
          daemon=True
        )
        t.start()
        threads.append(t)
    except Exception:
      cloudlog.exception(f"Error initializing {service} sensor")

  # Configure IRQ affinity (simplified)
  try:
    irq_path = "/proc/irq/336/smp_affinity_list"
    if not os.path.exists(irq_path):
      irq_path = "/proc/irq/335/smp_affinity_list"
    if os.path.exists(irq_path):
      with open(irq_path, 'w', encoding='utf-8') as f:
        f.write('1\n')
  except Exception:
    cloudlog.exception("Failed to set IRQ affinity")

  # Run interrupt loop
  t = threading.Thread(target=interrupt_loop, args=(sensors_cfg, exit_event), daemon=True)
  t.start()
  threads.append(t)

  try:
    while any(t.is_alive() for t in threads):
      time.sleep(1)
  except KeyboardInterrupt:
    exit_event.set()
  finally:
    cloudlog.warning("cleaning up")
    for t in threads:
      if t.is_alive():
        t.join()

    for sensor, _ in sensors_cfg:
      try:
        sensor.shutdown()
      except Exception:
        cloudlog.exception("Error shutting down sensor")

if __name__ == "__main__":
  main()
