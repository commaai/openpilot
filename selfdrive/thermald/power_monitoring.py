import random
import threading
import time
from statistics import mean
from typing import Optional

from cereal import log
from common.params import Params, put_nonblocking
from common.realtime import sec_since_boot
from selfdrive.hardware import HARDWARE
from selfdrive.swaglog import cloudlog

CAR_VOLTAGE_LOW_PASS_K = 0.091 # LPF gain for 5s tau (dt/tau / (dt/tau + 1))

# A C2 uses about 1W while idling, and 30h seens like a good shutoff for most cars
# While driving, a battery charges completely in about 30-60 minutes
CAR_BATTERY_CAPACITY_uWh = 30e6
CAR_CHARGING_RATE_W = 45

VBATT_PAUSE_CHARGING = 11.0           # Lower limit on the LPF car battery voltage
VBATT_INSTANT_PAUSE_CHARGING = 7.0    # Lower limit on the instant car battery voltage measurements to avoid triggering on instant power loss
MAX_TIME_OFFROAD_S = 30*3600
MIN_ON_TIME_S = 3600

class PowerMonitoring:
  def __init__(self):
    self.params = Params()
    self.last_measurement_time = None           # Used for integration delta
    self.last_save_time = 0                     # Used for saving current value in a param
    self.power_used_uWh = 0                     # Integrated power usage in uWh since going into offroad
    self.next_pulsed_measurement_time = None
    self.car_voltage_mV = 12e3                  # Low-passed version of peripheralState voltage
    self.car_voltage_instant_mV = 12e3          # Last value of peripheralState voltage
    self.integration_lock = threading.Lock()

    car_battery_capacity_uWh = self.params.get("CarBatteryCapacity")
    if car_battery_capacity_uWh is None:
      car_battery_capacity_uWh = 0

    # Reset capacity if it's low
    self.car_battery_capacity_uWh = max((CAR_BATTERY_CAPACITY_uWh / 10), int(car_battery_capacity_uWh))

  # Calculation tick
  def calculate(self, peripheralState, ignition):
    try:
      now = sec_since_boot()

      # If peripheralState is None, we're probably not in a car, so we don't care
      if peripheralState is None or peripheralState.pandaType == log.PandaState.PandaType.unknown:
        with self.integration_lock:
          self.last_measurement_time = None
          self.next_pulsed_measurement_time = None
          self.power_used_uWh = 0
        return

      # Low-pass battery voltage
      self.car_voltage_instant_mV = peripheralState.voltage
      self.car_voltage_mV = ((peripheralState.voltage * CAR_VOLTAGE_LOW_PASS_K) + (self.car_voltage_mV * (1 - CAR_VOLTAGE_LOW_PASS_K)))

      # Cap the car battery power and save it in a param every 10-ish seconds
      self.car_battery_capacity_uWh = max(self.car_battery_capacity_uWh, 0)
      self.car_battery_capacity_uWh = min(self.car_battery_capacity_uWh, CAR_BATTERY_CAPACITY_uWh)
      if now - self.last_save_time >= 10:
        put_nonblocking("CarBatteryCapacity", str(int(self.car_battery_capacity_uWh)))
        self.last_save_time = now

      # First measurement, set integration time
      with self.integration_lock:
        if self.last_measurement_time is None:
          self.last_measurement_time = now
          return

      if ignition:
        # If there is ignition, we integrate the charging rate of the car
        with self.integration_lock:
          self.power_used_uWh = 0
          integration_time_h = (now - self.last_measurement_time) / 3600
          if integration_time_h < 0:
            raise ValueError(f"Negative integration time: {integration_time_h}h")
          self.car_battery_capacity_uWh += (CAR_CHARGING_RATE_W * 1e6 * integration_time_h)
          self.last_measurement_time = now
      else:
        # No ignition, we integrate the offroad power used by the device
        is_uno = peripheralState.pandaType == log.PandaState.PandaType.uno
        # Get current power draw somehow
        current_power = HARDWARE.get_current_power_draw() # pylint: disable=assignment-from-none
        if current_power is not None:
          pass
        elif HARDWARE.get_battery_status() == 'Discharging':
          # If the battery is discharging, we can use this measurement
          # On C2: this is low by about 10-15%, probably mostly due to UNO draw not being factored in
          current_power = ((HARDWARE.get_battery_voltage() / 1000000) * (HARDWARE.get_battery_current() / 1000000))
        elif (self.next_pulsed_measurement_time is not None) and (self.next_pulsed_measurement_time <= now):
          # TODO: Figure out why this is off by a factor of 3/4???
          FUDGE_FACTOR = 1.33

          # Turn off charging for about 10 sec in a thread that does not get killed on SIGINT, and perform measurement here to avoid blocking thermal
          def perform_pulse_measurement(now):
            try:
              HARDWARE.set_battery_charging(False)
              time.sleep(5)

              # Measure for a few sec to get a good average
              voltages = []
              currents = []
              for _ in range(6):
                voltages.append(HARDWARE.get_battery_voltage())
                currents.append(HARDWARE.get_battery_current())
                time.sleep(1)
              current_power = ((mean(voltages) / 1000000) * (mean(currents) / 1000000))

              self._perform_integration(now, current_power * FUDGE_FACTOR)

              # Enable charging again
              HARDWARE.set_battery_charging(True)
            except Exception:
              cloudlog.exception("Pulsed power measurement failed")

          # Start pulsed measurement and return
          threading.Thread(target=perform_pulse_measurement, args=(now,)).start()
          self.next_pulsed_measurement_time = None
          return

        elif self.next_pulsed_measurement_time is None and not is_uno:
          # On a charging EON with black panda, or drawing more than 400mA out of a white/grey one
          # Only way to get the power draw is to turn off charging for a few sec and check what the discharging rate is
          # We shouldn't do this very often, so make sure it has been some long-ish random time interval
          self.next_pulsed_measurement_time = now + random.randint(120, 180)
          return
        else:
          # Do nothing
          return

        # Do the integration
        self._perform_integration(now, current_power)
    except Exception:
      cloudlog.exception("Power monitoring calculation failed")

  def _perform_integration(self, t: float, current_power: float) -> None:
    with self.integration_lock:
      try:
        if self.last_measurement_time:
          integration_time_h = (t - self.last_measurement_time) / 3600
          power_used = (current_power * 1000000) * integration_time_h
          if power_used < 0:
            raise ValueError(f"Negative power used! Integration time: {integration_time_h} h Current Power: {power_used} uWh")
          self.power_used_uWh += power_used
          self.car_battery_capacity_uWh -= power_used
          self.last_measurement_time = t
      except Exception:
        cloudlog.exception("Integration failed")

  # Get the power usage
  def get_power_used(self) -> int:
    return int(self.power_used_uWh)

  def get_car_battery_capacity(self) -> int:
    return int(self.car_battery_capacity_uWh)

  # See if we need to disable charging
  def should_disable_charging(self, ignition: bool, in_car: bool, offroad_timestamp: Optional[float]) -> bool:
    if offroad_timestamp is None:
      return False

    now = sec_since_boot()
    disable_charging = False
    disable_charging |= (now - offroad_timestamp) > MAX_TIME_OFFROAD_S
    disable_charging |= (self.car_voltage_mV < (VBATT_PAUSE_CHARGING * 1e3)) and (self.car_voltage_instant_mV > (VBATT_INSTANT_PAUSE_CHARGING * 1e3))
    disable_charging |= (self.car_battery_capacity_uWh <= 0)
    disable_charging &= not ignition
    disable_charging &= (not self.params.get_bool("DisablePowerDown"))
    disable_charging &= in_car
    disable_charging |= self.params.get_bool("ForcePowerDown")
    return disable_charging

  # See if we need to shutdown
  def should_shutdown(self, peripheralState, ignition, in_car, offroad_timestamp, started_seen):
    if offroad_timestamp is None:
      return False

    now = sec_since_boot()
    panda_charging = (peripheralState.usbPowerMode != log.PeripheralState.UsbPowerMode.client)
    BATT_PERC_OFF = 10

    should_shutdown = False
    # Wait until we have shut down charging before powering down
    should_shutdown |= (not panda_charging and self.should_disable_charging(ignition, in_car, offroad_timestamp))
    should_shutdown |= ((HARDWARE.get_battery_capacity() < BATT_PERC_OFF) and (not HARDWARE.get_battery_charging()) and ((now - offroad_timestamp) > 60))
    should_shutdown &= started_seen or (now > MIN_ON_TIME_S)
    return should_shutdown
