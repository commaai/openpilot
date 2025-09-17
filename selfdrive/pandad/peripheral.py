import os
import threading
import time
import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.hardware import HARDWARE

NO_FAN_CONTROL = os.getenv("NO_FAN_CONTROL") == "1"

MAX_IR_PANDA_VAL = 50
CUTOFF_IL = 400
SATURATE_IL = 1000


class HardwareReader:
  def __init__(self):
    self.voltage = 0
    self.current = 0
    self.running = True
    self.thread = threading.Thread(target=self._read_loop, daemon=True)
    self.thread.start()

  def _read_loop(self):
    while self.running:
      start = time.monotonic()
      try:
        self.voltage = HARDWARE.get_voltage()
        self.current = HARDWARE.get_current()
        elapsed = (time.monotonic() - start) * 1000
        if elapsed > 50:
          cloudlog.warning(f"hwmon read took {elapsed:.2f}ms")
      except Exception as e:
        cloudlog.error(f"Hardware read error: {e}")
      time.sleep(0.5)  # 500ms update rate

  def get_values(self):
    return self.voltage, self.current

  def stop(self):
    self.running = False
    self.thread.join()


class PeripheralManager:
  def __init__(self, panda, hw_type):
    self.panda = panda
    self.last_camera_t = 0
    self.prev_fan = 999
    self.prev_ir_pwr = 999
    self.ir_pwr = 0
    self.filter = FirstOrderFilter(0, 30.0, 0.05)
    self.hw_type = hw_type
    self.hw_reader = HardwareReader()

  def process(self, sm):
    if sm.updated["deviceState"] and not NO_FAN_CONTROL:
      fan = sm["deviceState"].fanSpeedPercentDesired
      if fan != self.prev_fan or sm.frame % 100 == 0:
        self.panda.set_fan_power(fan)
        self.prev_fan = fan

    if sm.updated["driverCameraState"]:
      state = sm["driverCameraState"]
      lines = self.filter.update(state.integLines)
      self.last_camera_t = sm.logMonoTime['driverCameraState']
      if lines <= CUTOFF_IL:
        self.ir_pwr = 0
      elif lines > SATURATE_IL:
        self.ir_pwr = 100
      else:
        self.ir_pwr = 100 * (lines - CUTOFF_IL) / (SATURATE_IL - CUTOFF_IL)

    if time.monotonic_ns() - self.last_camera_t > 1e9:
      self.ir_pwr = 0

    if self.ir_pwr != self.prev_ir_pwr or sm.frame % 100 == 0:
      ir_panda = int(self.ir_pwr * MAX_IR_PANDA_VAL / 100)
      self.panda.set_ir_power(ir_panda)

      HARDWARE.set_ir_power(self.ir_pwr)
      self.prev_ir_pwr = self.ir_pwr

  def send_state(self, pm):
    msg = messaging.new_message('peripheralState')
    msg.valid = True
    ps = msg.peripheralState
    ps.pandaType = self.hw_type
    ps.voltage, ps.current = self.hw_reader.get_values()

    if not (ps.voltage or ps.current):
      h = self.panda.health() or {}
      ps.voltage = h.get("voltage", 0)
      ps.current = h.get("current", 0)

    ps.fanSpeedRpm = self.panda.get_fan_rpm()

    pm.send("peripheralState", msg)

  def cleanup(self):
    self.hw_reader.stop()
