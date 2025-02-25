import os
import time
import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from openpilot.common.filter_simple import FirstOrderFilter

NO_FAN_CONTROL = os.getenv("NO_FAN_CONTROL") == "1"

MAX_IR_POWER = 0.5
MIN_IR_POWER = 0.0
CUTOFF_IL = 400
SATURATE_IL = 1000

def get_voltage():
  with open("/sys/class/hwmon/hwmon1/in1_input") as f:
    return int(f.read())

def get_current():
  with open("/sys/class/hwmon/hwmon1/curr1_input") as f:
    return int(f.read())


class PeripheralManager:
  def __init__(self, pandas, lock):
    self.panda = pandas[0]
    self.last_camera_t = 0
    self.prev_fan = 999
    self.prev_ir = 999
    self.filter = FirstOrderFilter(0, 30.0, 0.05)
    self.lock = lock

  def process(self, sm):
    if sm.updated["deviceState"] and not NO_FAN_CONTROL:
      fan = sm["deviceState"].fanSpeedPercentDesired
      if fan != self.prev_fan or sm.frame % 100 == 0:
        with self.lock:
          self.panda.set_fan_power(fan)
        self.prev_fan = fan

    ir = None
    if sm.updated["driverCameraState"]:
      state = sm["driverCameraState"]
      print(state)
      lines = self.filter.update(state.integLines)
      self.last_camera_t = sm.logMonoTime['driverCameraState']
      if lines <= CUTOFF_IL:
        ir = 100.0 * MIN_IR_POWER
      elif lines > SATURATE_IL:
        ir = 100.0 * MAX_IR_POWER
      else:
        slope = (MAX_IR_POWER - MIN_IR_POWER) / (SATURATE_IL - CUTOFF_IL)
        ir = 100.0 * (MIN_IR_POWER + slope * (lines - CUTOFF_IL))

    if time.monotonic_ns() - self.last_camera_t > 1e9:
      ir = 0
    if ir and (ir != self.prev_ir or sm.frame % 100 == 0 or ir >= 50):
      with self.lock:
        self.panda.set_ir_power(ir)
      self.prev_ir = ir

  def send_state(self, pm):
    msg = messaging.new_message('peripheralState')
    ps = msg.peripheralState

    start = time.monotonic()
    ps.voltage = get_voltage()
    ps.current = get_current()
    if (time.monotonic() - start) * 1000 > 50:
      cloudlog.warning(f"hwmon read took {(time.monotonic() - start) * 1000:.2f}ms")

    if not (ps.voltage or ps.current):
      with self.lock:
        h = self.panda.health() or {}
      ps.voltage = h.get("voltage", 0)
      ps.current = h.get("current", 0)

    with self.lock:
      ps.fanSpeedRpm = self.panda.get_fan_rpm()

    pm.send("peripheralState", msg)
