import time
import datetime
import threading
import random
from statistics import mean
from cereal import log

PANDA_OUTPUT_VOLTAGE = 5.28

# Helpers
def _read_param(path, parser, default=0):
  try:
    with open(path) as f:
      return parser(f.read())
  except FileNotFoundError:
    return default

def panda_current_to_actual_current(panda_current):
  # From white/grey panda schematic
  return (3.3 - (panda_current * 3.3 / 4096)) / 8.25

# Parameters
def get_battery_capacity():
  return _read_param("/sys/class/power_supply/battery/capacity", int)

def get_battery_status():
  # This does not correspond with actual charging or not. 
  # If a USB cable is plugged in, it responds with 'Charging', even when charging is disabled
  return _read_param("/sys/class/power_supply/battery/status", lambda x: x.strip(), '')

def get_battery_current():
  return _read_param("/sys/class/power_supply/battery/current_now", int)

def get_battery_voltage():
  return _read_param("/sys/class/power_supply/battery/voltage_now", int)

def get_usb_present():
  return _read_param("/sys/class/power_supply/usb/present", lambda x: bool(int(x)), False)

def get_battery_charging():
  # This does correspond with actually charging
  return _read_param("/sys/class/power_supply/battery/charge_type", lambda x: x.strip() != "N/A", False)

def set_battery_charging(on):
  with open('/sys/class/power_supply/battery/charging_enabled', 'w') as f:
    f.write(f"{1 if on else 0}\n")

last_measurement_time = None           # Used for integration delta
power_used_uWh = 0                    # Integrated power usage in uWh since going into offroad
next_pulsed_measurement_time = None

# Calculation tick
def pm_calculate(health):
  global power_used_uWh, last_measurement_time, next_pulsed_measurement_time
  now = time.time()

  # Check that time is valid
  if datetime.datetime.fromtimestamp(now).year < 2019:
    return

  # Only integrate when there is no ignition
  # If health is None, we're probably not in a car, so we don't care
  if health == None or (health.health.ignitionLine or health.health.ignitionCan):
    last_measurement_time = None
    power_used_uWh = 0
    return

  # First measurement, set integration time
  if last_measurement_time = None:
    last_measurement_time = now
    return

  # Get current power draw somehow
  current_power = 0
  if get_battery_status() == 'Discharging':
    # If the battery is discharging, we can use this measurement
    current_power = ((get_battery_voltage() / 1000000)  * (get_battery_current() / 1000000))
  elif (health.health.hwType in [log.HealthData.HwType.whitePanda, log.HealthData.HwType.greyPanda]) and (health.health.current > 1):
    # If white/grey panda, use the integrated current measurements if the measurement is not 0
    # If the measurement is 0, the current is 400mA or greater, and out of the measurement range of the panda
    current_power = (PANDA_OUTPUT_VOLTAGE * panda_current_to_actual_current(health.health.current))
  elif (next_pulsed_measurement_time != None) and (next_pulsed_measurement_time <= now):
    # Turn off charging for 10 sec in a thread that does not get killed on SIGINT
    def charging_temp_disable():
      set_battery_charging(False)
      time.sleep(10)
      set_battery_charging(True)
    threading.Thread(target=charging_temp_disable).start()
    time.sleep(1)

    # Measure for a few sec to get a good average
    voltages = []
    currents = []
    for i in range(6):
      voltages.append(get_battery_voltage())
      currents.append(get_battery_current())
      time.sleep(1)
    current_power = ((mean(voltages) / 1000000)  * (mean(currents) / 1000000))
    next_pulsed_measurement_time = None
  elif next_pulsed_measurement_time == None:
    # On a charging EON with black panda, or drawing more than 400mA out of a white/grey one
    # Only way to get the power draw is to turn off charging for a few sec and check what the discharging rate is
    # We shouldn't do this very often, so make sure it has been some long-ish random time interval
    next_pulsed_measurement_time = now + random.randint(120, 180)
    return
  else:
    # Do nothing
    return

  # Do the integration
  integration_time_h = (now - last_measurement_time) / 3600
  power_used_uWh += (current_power * 1000000) * integration_time_h
  last_measurement_time = now

# Get the power usage
def get_power_used():
  return power_used_uWh




