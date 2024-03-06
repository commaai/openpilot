#!/usr/bin/env python3
import unittest
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.selfdrive.thermald.power_monitoring import PowerMonitoring, CAR_BATTERY_CAPACITY_uWh, \
                                                CAR_CHARGING_RATE_W, VBATT_PAUSE_CHARGING, DELAY_SHUTDOWN_TIME_S


# Create fake time
ssb = 0.
def mock_time_monotonic():
  global ssb
  ssb += 1.
  return ssb

TEST_DURATION_S = 50
GOOD_VOLTAGE = 12 * 1e3
VOLTAGE_BELOW_PAUSE_CHARGING = (VBATT_PAUSE_CHARGING - 1) * 1e3

def pm_patch(name, value, constant=False):
  if constant:
    return patch(f"openpilot.selfdrive.thermald.power_monitoring.{name}", value)
  return patch(f"openpilot.selfdrive.thermald.power_monitoring.{name}", return_value=value)


@patch("time.monotonic", new=mock_time_monotonic)
class TestPowerMonitoring(unittest.TestCase):
  def setUp(self):
    self.params = Params()

  # Test to see that it doesn't do anything when pandaState is None
  def test_pandaState_present(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(None, None)
    self.assertEqual(pm.get_power_used(), 0)
    self.assertEqual(pm.get_car_battery_capacity(), (CAR_BATTERY_CAPACITY_uWh / 10))

  # Test to see that it doesn't integrate offroad when ignition is True
  def test_offroad_ignition(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(GOOD_VOLTAGE, True)
    self.assertEqual(pm.get_power_used(), 0)

  # Test to see that it integrates with discharging battery
  def test_offroad_integration_discharging(self):
    POWER_DRAW = 4
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(GOOD_VOLTAGE, False)
      expected_power_usage = ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
      self.assertLess(abs(pm.get_power_used() - expected_power_usage), 10)

  # Test to check positive integration of car_battery_capacity
  def test_car_battery_integration_onroad(self):
    POWER_DRAW = 4
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = 0
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(GOOD_VOLTAGE, True)
      expected_capacity = ((TEST_DURATION_S/3600) * CAR_CHARGING_RATE_W * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - expected_capacity), 10)

  # Test to check positive integration upper limit
  def test_car_battery_integration_upper_limit(self):
    POWER_DRAW = 4
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh - 1000
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(GOOD_VOLTAGE, True)
      estimated_capacity = CAR_BATTERY_CAPACITY_uWh + (CAR_CHARGING_RATE_W / 3600 * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - estimated_capacity), 10)

  # Test to check negative integration of car_battery_capacity
  def test_car_battery_integration_offroad(self):
    POWER_DRAW = 4
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(GOOD_VOLTAGE, False)
      expected_capacity = CAR_BATTERY_CAPACITY_uWh - ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - expected_capacity), 10)

  # Test to check negative integration lower limit
  def test_car_battery_integration_lower_limit(self):
    POWER_DRAW = 4
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = 1000
      for _ in range(TEST_DURATION_S + 1):
        pm.calculate(GOOD_VOLTAGE, False)
      estimated_capacity = 0 - ((1/3600) * POWER_DRAW * 1e6)
      self.assertLess(abs(pm.get_car_battery_capacity() - estimated_capacity), 10)

  # Test to check policy of stopping charging after MAX_TIME_OFFROAD_S
  def test_max_time_offroad(self):
    MOCKED_MAX_OFFROAD_TIME = 3600
    POWER_DRAW = 0 # To stop shutting down for other reasons
    with pm_patch("MAX_TIME_OFFROAD_S", MOCKED_MAX_OFFROAD_TIME, constant=True), pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      start_time = ssb
      ignition = False
      while ssb <= start_time + MOCKED_MAX_OFFROAD_TIME:
        pm.calculate(GOOD_VOLTAGE, ignition)
        if (ssb - start_time) % 1000 == 0 and ssb < start_time + MOCKED_MAX_OFFROAD_TIME:
          self.assertFalse(pm.should_shutdown(ignition, True, start_time, False))
      self.assertTrue(pm.should_shutdown(ignition, True, start_time, False))

  def test_car_voltage(self):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 350
    VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S = 50
    with pm_patch("VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S", VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S, constant=True), \
         pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      ignition = False
      start_time = ssb
      for i in range(TEST_TIME):
        pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
        if i % 10 == 0:
          self.assertEqual(pm.should_shutdown(ignition, True, start_time, True),
                           (pm.car_voltage_mV < VBATT_PAUSE_CHARGING * 1e3 and
                            (ssb - start_time) > VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S and
                             (ssb - start_time) > DELAY_SHUTDOWN_TIME_S))
      self.assertTrue(pm.should_shutdown(ignition, True, start_time, True))

  # Test to check policy of not stopping charging when DisablePowerDown is set
  def test_disable_power_down(self):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    self.params.put_bool("DisablePowerDown", True)
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      ignition = False
      for i in range(TEST_TIME):
        pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
        if i % 10 == 0:
          self.assertFalse(pm.should_shutdown(ignition, True, ssb, False))
      self.assertFalse(pm.should_shutdown(ignition, True, ssb, False))

  # Test to check policy of not stopping charging when ignition
  def test_ignition(self):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
      ignition = True
      for i in range(TEST_TIME):
        pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
        if i % 10 == 0:
          self.assertFalse(pm.should_shutdown(ignition, True, ssb, False))
      self.assertFalse(pm.should_shutdown(ignition, True, ssb, False))

  # Test to check policy of not stopping charging when harness is not connected
  def test_harness_connection(self):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    with pm_patch("HARDWARE.get_current_power_draw", POWER_DRAW):
      pm = PowerMonitoring()
      pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

      ignition = False
      for i in range(TEST_TIME):
        pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
        if i % 10 == 0:
          self.assertFalse(pm.should_shutdown(ignition, False, ssb, False))
      self.assertFalse(pm.should_shutdown(ignition, False, ssb, False))

  def test_delay_shutdown_time(self):
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 0
    ignition = False
    in_car = True
    offroad_timestamp = ssb
    started_seen = True
    pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)

    while ssb < offroad_timestamp + DELAY_SHUTDOWN_TIME_S:
      self.assertFalse(pm.should_shutdown(ignition, in_car,
                                          offroad_timestamp,
                                          started_seen),
                       f"Should not shutdown before {DELAY_SHUTDOWN_TIME_S} seconds offroad time")
    self.assertTrue(pm.should_shutdown(ignition, in_car,
                                       offroad_timestamp,
                                       started_seen),
                    f"Should shutdown after {DELAY_SHUTDOWN_TIME_S} seconds offroad time")


if __name__ == "__main__":
  unittest.main()
