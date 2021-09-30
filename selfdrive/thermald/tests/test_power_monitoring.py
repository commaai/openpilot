#!/usr/bin/env python3
import unittest
from unittest.mock import patch
from parameterized import parameterized

from cereal import log
import cereal.messaging as messaging
from common.params import Params
params = Params()

# Create fake time
ssb = 0
def mock_sec_since_boot():
  global ssb
  ssb += 1
  return ssb

with patch("common.realtime.sec_since_boot", new=mock_sec_since_boot):
  with patch("common.params.put_nonblocking", new=params.put):
    from selfdrive.thermald.power_monitoring import PowerMonitoring, CAR_BATTERY_CAPACITY_uWh, \
                                                    CAR_CHARGING_RATE_W, VBATT_PAUSE_CHARGING

TEST_DURATION_S = 50
ALL_PANDA_TYPES = [(hw_type,) for hw_type in [log.PandaState.PandaType.whitePanda,
                                              log.PandaState.PandaType.greyPanda,
                                              log.PandaState.PandaType.blackPanda,
                                              log.PandaState.PandaType.uno]]

def pm_patch(name, value, constant=False):
  if constant:
    return patch(f"selfdrive.thermald.power_monitoring.{name}", value)
  return patch(f"selfdrive.thermald.power_monitoring.{name}", return_value=value)

class TestPowerMonitoring(unittest.TestCase):
  def setUp(self):
    # Clear stored capacity before each test
    params.delete("CarBatteryCapacity")
    params.delete("DisablePowerDown")

  def mock_peripheralState(self, hw_type, car_voltage=12,):
    peripheralState = messaging.new_message('peripheralState')
    peripheralState.peripheralState.pandaType = hw_type
    peripheralState.peripheralState.voltage = car_voltage * 1e3
    return peripheralState.peripheralState

  # Test to see that it doesn't do anything when pandaState is None
  def test_pandaState_present(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(None, None)
    self.assertEqual(pm.get_power_used(), 0)
    self.assertEqual(pm.get_car_battery_capacity(), (CAR_BATTERY_CAPACITY_uWh / 10))

  # Test to see that it doesn't integrate offroad when ignition is True
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_offroad_ignition(self, hw_type):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(self.mock_peripheralState(hw_type), True)
    self.assertEqual(pm.get_power_used(), 0)

  # Test to see that it integrates with discharging battery
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_offroad_integration_discharging(self, hw_type):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 1
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(self.mock_peripheralState(hw_type), False)
      expected_power_usage = ((TEST_DURATION_S/3600) * (BATT_VOLTAGE * BATT_CURRENT) * 1e6)
      self.assertLess(abs(pm.get_power_used() - expected_power_usage), 10)

  # Test to check positive integration of car_battery_capacity
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_car_battery_integration_onroad(self, hw_type):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 1
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = 0
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(self.mock_peripheralState(hw_type), True)
      expected_capacity = ((TEST_DURATION_S/3600) * CAR_CHARGING_RATE_W * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - expected_capacity), 10)

  # Test to check positive integration upper limit
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_car_battery_integration_upper_limit(self, hw_type):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 1
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh - 1000
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(self.mock_peripheralState(hw_type), True)
      estimated_capacity = CAR_BATTERY_CAPACITY_uWh + (CAR_CHARGING_RATE_W / 3600 * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - estimated_capacity), 10)

  # Test to check negative integration of car_battery_capacity
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_car_battery_integration_offroad(self, hw_type):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 1
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(self.mock_peripheralState(hw_type), False)
      expected_capacity = CAR_BATTERY_CAPACITY_uWh - ((TEST_DURATION_S/3600) * (BATT_VOLTAGE * BATT_CURRENT) * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - expected_capacity), 10)

  # Test to check negative integration lower limit
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_car_battery_integration_lower_limit(self, hw_type):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 1
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = 1000
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(self.mock_peripheralState(hw_type), False)
      estimated_capacity = 0 - ((1/3600) * (BATT_VOLTAGE * BATT_CURRENT) * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - estimated_capacity), 10)

  # Test to check policy of stopping charging after MAX_TIME_OFFROAD_S
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_max_time_offroad(self, hw_type):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 0 # To stop shutting down for other reasons
    MOCKED_MAX_OFFROAD_TIME = 3600
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("MAX_TIME_OFFROAD_S", MOCKED_MAX_OFFROAD_TIME, constant=True), \
    pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      start_time = ssb
      ignition = False
      peripheralState = self.mock_peripheralState(hw_type)
      while ssb <= start_time + MOCKED_MAX_OFFROAD_TIME:
        pm.calculate(peripheralState, ignition)
        if (ssb - start_time) % 1000 == 0 and ssb < start_time + MOCKED_MAX_OFFROAD_TIME:
          self.assertFalse(pm.should_disable_charging(ignition, start_time))
      self.assertTrue(pm.should_disable_charging(ignition, start_time))

  # Test to check policy of stopping charging when the car voltage is too low
  @parameterized.expand(ALL_PANDA_TYPES)
  def test_car_voltage(self, hw_type):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      ignition = False
      peripheralState = self.mock_peripheralState(hw_type, car_voltage=(VBATT_PAUSE_CHARGING - 1))
      for i in range(TEST_TIME):
        pm.calculate(peripheralState, ignition)
        if i % 10 == 0:
          self.assertEqual(pm.should_disable_charging(ignition, ssb), (pm.car_voltage_mV < VBATT_PAUSE_CHARGING*1e3))
      self.assertTrue(pm.should_disable_charging(ignition, ssb))

  # Test to check policy of not stopping charging when DisablePowerDown is set
  def test_disable_power_down(self):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    params.put_bool("DisablePowerDown", True)
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      ignition = False
      peripheralState = self.mock_peripheralState(log.PandaState.PandaType.uno, car_voltage=(VBATT_PAUSE_CHARGING - 1))
      for i in range(TEST_TIME):
        pm.calculate(peripheralState, ignition)
        if i % 10 == 0:
          self.assertFalse(pm.should_disable_charging(ignition, ssb))
      self.assertFalse(pm.should_disable_charging(ignition, ssb))

  # Test to check policy of not stopping charging when ignition
  def test_ignition(self):
    BATT_VOLTAGE = 4
    BATT_CURRENT = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    with pm_patch("HARDWARE.get_battery_voltage", BATT_VOLTAGE * 1e6), pm_patch("HARDWARE.get_battery_current", BATT_CURRENT * 1e6), \
    pm_patch("HARDWARE.get_battery_status", "Discharging"), pm_patch("HARDWARE.get_current_power_draw", None):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      ignition = True
      peripheralState = self.mock_peripheralState(log.PandaState.PandaType.uno, car_voltage=(VBATT_PAUSE_CHARGING - 1))
      for i in range(TEST_TIME):
        pm.calculate(peripheralState, ignition)
        if i % 10 == 0:
          self.assertFalse(pm.should_disable_charging(ignition, ssb))
      self.assertFalse(pm.should_disable_charging(ignition, ssb))


if __name__ == "__main__":
  unittest.main()
