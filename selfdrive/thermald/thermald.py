#!/usr/bin/env python3
import datetime
import os
import time
from collections import namedtuple

import psutil
from smbus2 import SMBus

import cereal.messaging as messaging
from cereal import log
from common.android import get_network_strength, get_network_type
from common.filter_simple import FirstOrderFilter
from common.numpy_fast import clip, interp
from common.params import Params, put_nonblocking
from common.realtime import DT_TRML, sec_since_boot
from selfdrive.controls.lib.alertmanager import set_offroad_alert
from selfdrive.loggerd.config import get_available_percent
from selfdrive.pandad import get_expected_signature
from selfdrive.swaglog import cloudlog
from selfdrive.thermald.power_monitoring import (PowerMonitoring,
                                                 get_battery_capacity,
                                                 get_battery_current,
                                                 get_battery_status,
                                                 get_battery_voltage,
                                                 get_usb_present)
from selfdrive.version import get_git_branch, terms_version, training_version

ThermalConfig = namedtuple('ThermalConfig', ['cpu', 'gpu', 'mem', 'bat', 'ambient'])

FW_SIGNATURE = get_expected_signature()

ThermalStatus = log.ThermalData.ThermalStatus
NetworkType = log.ThermalData.NetworkType
NetworkStrength = log.ThermalData.NetworkStrength
CURRENT_TAU = 15.   # 15s time constant
CPU_TEMP_TAU = 5.   # 5s time constant
DAYS_NO_CONNECTIVITY_MAX = 7  # do not allow to engage after a week without internet
DAYS_NO_CONNECTIVITY_PROMPT = 4  # send an offroad prompt after 4 days with no internet
DISCONNECT_TIMEOUT = 5.  # wait 5 seconds before going offroad after disconnect so you get an alert

EON = os.path.isfile('/EON')
TICI = os.path.isfile('/TICI')

LEON = False
last_eon_fan_val = None


def get_thermal_config():
  # (tz, scale)
  if EON:
    return ThermalConfig(cpu=((5, 7, 10, 12), 10), gpu=((16,), 10), mem=(2, 10), bat=(29, 1000), ambient=(25, 1))
  elif TICI:
    return ThermalConfig(cpu=((1, 2, 3, 4, 5, 6, 7, 8), 1000), gpu=((48,49), 1000), mem=(15, 1000), bat=(None, 1), abient=(70, 1000))
  else:
    return ThermalConfig(cpu=((None,), 1), gpu=((None,), 1), mem=(None, 1), bat=(None, 1), ambient=(None, 1))


def read_tz(x):
  if x is None:
    return 0

  try:
    with open("/sys/devices/virtual/thermal/thermal_zone%d/temp" % x) as f:
      return int(f.read())
  except FileNotFoundError:
    return 0


def read_thermal(thermal_config):
  dat = messaging.new_message('thermal')
  dat.thermal.cpu = [read_tz(z) / thermal_config.cpu[1] for z in thermal_config.cpu[0]]
  dat.thermal.gpu = [read_tz(z) / thermal_config.gpu[1] for z in thermal_config.gpu[0]]
  dat.thermal.mem = read_tz(thermal_config.mem[0]) / thermal_config.mem[1]
  dat.thermal.ambient = read_tz(thermal_config.ambient[0]) / thermal_config.ambient[1]
  dat.thermal.bat = read_tz(thermal_config.bat[0]) / thermal_config.bat[1]
  return dat


def setup_eon_fan():
  global LEON

  os.system("echo 2 > /sys/module/dwc3_msm/parameters/otg_switch")

  bus = SMBus(7, force=True)
  try:
    bus.write_byte_data(0x21, 0x10, 0xf)   # mask all interrupts
    bus.write_byte_data(0x21, 0x03, 0x1)   # set drive current and global interrupt disable
    bus.write_byte_data(0x21, 0x02, 0x2)   # needed?
    bus.write_byte_data(0x21, 0x04, 0x4)   # manual override source
  except IOError:
    print("LEON detected")
    LEON = True
  bus.close()


def set_eon_fan(val):
  global LEON, last_eon_fan_val

  if last_eon_fan_val is None or last_eon_fan_val != val:
    bus = SMBus(7, force=True)
    if LEON:
      try:
        i = [0x1, 0x3 | 0, 0x3 | 0x08, 0x3 | 0x10][val]
        bus.write_i2c_block_data(0x3d, 0, [i])
      except IOError:
        # tusb320
        if val == 0:
          bus.write_i2c_block_data(0x67, 0xa, [0])
          #bus.write_i2c_block_data(0x67, 0x45, [1<<2])
        else:
          #bus.write_i2c_block_data(0x67, 0x45, [0])
          bus.write_i2c_block_data(0x67, 0xa, [0x20])
          bus.write_i2c_block_data(0x67, 0x8, [(val - 1) << 6])
    else:
      bus.write_byte_data(0x21, 0x04, 0x2)
      bus.write_byte_data(0x21, 0x03, (val*2)+1)
      bus.write_byte_data(0x21, 0x04, 0x4)
    bus.close()
    last_eon_fan_val = val


# temp thresholds to control fan speed - high hysteresis
_TEMP_THRS_H = [50., 65., 80., 10000]
# temp thresholds to control fan speed - low hysteresis
_TEMP_THRS_L = [42.5, 57.5, 72.5, 10000]
# fan speed options
_FAN_SPEEDS = [0, 16384, 32768, 65535]
# max fan speed only allowed if battery is hot
_BAT_TEMP_THERSHOLD = 45.


def handle_fan_eon(max_cpu_temp, bat_temp, fan_speed, ignition):
  new_speed_h = next(speed for speed, temp_h in zip(_FAN_SPEEDS, _TEMP_THRS_H) if temp_h > max_cpu_temp)
  new_speed_l = next(speed for speed, temp_l in zip(_FAN_SPEEDS, _TEMP_THRS_L) if temp_l > max_cpu_temp)

  if new_speed_h > fan_speed:
    # update speed if using the high thresholds results in fan speed increment
    fan_speed = new_speed_h
  elif new_speed_l < fan_speed:
    # update speed if using the low thresholds results in fan speed decrement
    fan_speed = new_speed_l

  if bat_temp < _BAT_TEMP_THERSHOLD:
    # no max fan speed unless battery is hot
    fan_speed = min(fan_speed, _FAN_SPEEDS[-2])

  set_eon_fan(fan_speed // 16384)

  return fan_speed


def handle_fan_uno(max_cpu_temp, bat_temp, fan_speed, ignition):
  new_speed = int(interp(max_cpu_temp, [40.0, 80.0], [0, 80]))

  if not ignition:
    new_speed = min(30, new_speed)

  return new_speed


def thermald_thread():
  health_timeout = int(1000 * 2.5 * DT_TRML)  # 2.5x the expected health frequency

  # now loop
  thermal_sock = messaging.pub_sock('thermal')
  health_sock = messaging.sub_sock('health', timeout=health_timeout)
  location_sock = messaging.sub_sock('gpsLocation')

  ignition = False
  fan_speed = 0
  count = 0

  off_ts = None
  started_ts = None
  started_seen = False
  thermal_status = ThermalStatus.green
  thermal_status_prev = ThermalStatus.green
  usb_power = True
  usb_power_prev = True
  current_branch = get_git_branch()

  network_type = NetworkType.none
  network_strength = NetworkStrength.unknown

  current_filter = FirstOrderFilter(0., CURRENT_TAU, DT_TRML)
  cpu_temp_filter = FirstOrderFilter(0., CPU_TEMP_TAU, DT_TRML)
  health_prev = None
  fw_version_match_prev = True
  current_update_alert = None
  time_valid_prev = True
  should_start_prev = False
  handle_fan = None
  is_uno = False
  has_relay = False

  params = Params()
  pm = PowerMonitoring()
  no_panda_cnt = 0

  thermal_config = get_thermal_config()

  while 1:
    health = messaging.recv_sock(health_sock, wait=True)
    location = messaging.recv_sock(location_sock)
    location = location.gpsLocation if location else None
    msg = read_thermal(thermal_config)

    if health is not None:
      usb_power = health.health.usbPowerMode != log.HealthData.UsbPowerMode.client

      # If we lose connection to the panda, wait 5 seconds before going offroad
      if health.health.hwType == log.HealthData.HwType.unknown:
        no_panda_cnt += 1
        if no_panda_cnt > DISCONNECT_TIMEOUT / DT_TRML:
          if ignition:
            cloudlog.error("Lost panda connection while onroad")
          ignition = False
      else:
        no_panda_cnt = 0
        ignition = health.health.ignitionLine or health.health.ignitionCan

      # Setup fan handler on first connect to panda
      if handle_fan is None and health.health.hwType != log.HealthData.HwType.unknown:
        is_uno = health.health.hwType == log.HealthData.HwType.uno
        has_relay = health.health.hwType in [log.HealthData.HwType.blackPanda, log.HealthData.HwType.uno, log.HealthData.HwType.dos]

        if (not EON) or is_uno:
          cloudlog.info("Setting up UNO fan handler")
          handle_fan = handle_fan_uno
        else:
          cloudlog.info("Setting up EON fan handler")
          setup_eon_fan()
          handle_fan = handle_fan_eon

      # Handle disconnect
      if health_prev is not None:
        if health.health.hwType == log.HealthData.HwType.unknown and \
          health_prev.health.hwType != log.HealthData.HwType.unknown:
          params.panda_disconnect()
      health_prev = health

    # get_network_type is an expensive call. update every 10s
    if (count % int(10. / DT_TRML)) == 0:
      try:
        network_type = get_network_type()
        network_strength = get_network_strength(network_type)
      except Exception:
        cloudlog.exception("Error getting network status")

    msg.thermal.freeSpace = get_available_percent(default=100.0) / 100.0
    msg.thermal.memUsedPercent = int(round(psutil.virtual_memory().percent))
    msg.thermal.cpuPerc = int(round(psutil.cpu_percent()))
    msg.thermal.networkType = network_type
    msg.thermal.networkStrength = network_strength
    msg.thermal.batteryPercent = get_battery_capacity()
    msg.thermal.batteryStatus = get_battery_status()
    msg.thermal.batteryCurrent = get_battery_current()
    msg.thermal.batteryVoltage = get_battery_voltage()
    msg.thermal.usbOnline = get_usb_present()

    # Fake battery levels on uno for frame
    if (not EON) or is_uno:
      msg.thermal.batteryPercent = 100
      msg.thermal.batteryStatus = "Charging"
      msg.thermal.bat = 0

    current_filter.update(msg.thermal.batteryCurrent / 1e6)

    # TODO: add car battery voltage check
    max_cpu_temp = cpu_temp_filter.update(max(msg.thermal.cpu))
    max_comp_temp = max(max_cpu_temp, msg.thermal.mem, max(msg.thermal.gpu))
    bat_temp = msg.thermal.bat

    if handle_fan is not None:
      fan_speed = handle_fan(max_cpu_temp, bat_temp, fan_speed, ignition)
      msg.thermal.fanSpeed = fan_speed

    # If device is offroad we want to cool down before going onroad
    # since going onroad increases load and can make temps go over 107
    # We only do this if there is a relay that prevents the car from faulting
    if max_cpu_temp > 107. or bat_temp >= 63. or (has_relay and (started_ts is None) and max_cpu_temp > 70.0):
      # onroad not allowed
      thermal_status = ThermalStatus.danger
    elif max_comp_temp > 96.0 or bat_temp > 60.:
      # hysteresis between onroad not allowed and engage not allowed
      thermal_status = clip(thermal_status, ThermalStatus.red, ThermalStatus.danger)
    elif max_cpu_temp > 94.0:
      # hysteresis between engage not allowed and uploader not allowed
      thermal_status = clip(thermal_status, ThermalStatus.yellow, ThermalStatus.red)
    elif max_cpu_temp > 80.0:
      # uploader not allowed
      thermal_status = ThermalStatus.yellow
    elif max_cpu_temp > 75.0:
      # hysteresis between uploader not allowed and all good
      thermal_status = clip(thermal_status, ThermalStatus.green, ThermalStatus.yellow)
    else:
      # all good
      thermal_status = ThermalStatus.green

    # **** starting logic ****

    # Check for last update time and display alerts if needed
    now = datetime.datetime.utcnow()

    # show invalid date/time alert
    time_valid = now.year >= 2019
    if time_valid and not time_valid_prev:
      set_offroad_alert("Offroad_InvalidTime", False)
    if not time_valid and time_valid_prev:
      set_offroad_alert("Offroad_InvalidTime", True)
    time_valid_prev = time_valid

    # Show update prompt
    try:
      last_update = datetime.datetime.fromisoformat(params.get("LastUpdateTime", encoding='utf8'))
    except (TypeError, ValueError):
      last_update = now
    dt = now - last_update

    update_failed_count = params.get("UpdateFailedCount")
    update_failed_count = 0 if update_failed_count is None else int(update_failed_count)
    last_update_exception = params.get("LastUpdateException", encoding='utf8')

    if update_failed_count > 15 and last_update_exception is not None:
      if current_branch in ["release2", "dashcam"]:
        extra_text = "Ensure the software is correctly installed"
      else:
        extra_text = last_update_exception

      if current_update_alert != "update" + extra_text:
        current_update_alert = "update" + extra_text
        set_offroad_alert("Offroad_ConnectivityNeeded", False)
        set_offroad_alert("Offroad_ConnectivityNeededPrompt", False)
        set_offroad_alert("Offroad_UpdateFailed", True, extra_text=extra_text)
    elif dt.days > DAYS_NO_CONNECTIVITY_MAX and update_failed_count > 1:
      if current_update_alert != "expired":
        current_update_alert = "expired"
        set_offroad_alert("Offroad_UpdateFailed", False)
        set_offroad_alert("Offroad_ConnectivityNeededPrompt", False)
        set_offroad_alert("Offroad_ConnectivityNeeded", True)
    elif dt.days > DAYS_NO_CONNECTIVITY_PROMPT:
      remaining_time = str(max(DAYS_NO_CONNECTIVITY_MAX - dt.days, 0))
      if current_update_alert != "prompt" + remaining_time:
        current_update_alert = "prompt" + remaining_time
        set_offroad_alert("Offroad_UpdateFailed", False)
        set_offroad_alert("Offroad_ConnectivityNeeded", False)
        set_offroad_alert("Offroad_ConnectivityNeededPrompt", True, extra_text=f"{remaining_time} days.")
    elif current_update_alert is not None:
      current_update_alert = None
      set_offroad_alert("Offroad_UpdateFailed", False)
      set_offroad_alert("Offroad_ConnectivityNeeded", False)
      set_offroad_alert("Offroad_ConnectivityNeededPrompt", False)

    do_uninstall = params.get("DoUninstall") == b"1"
    accepted_terms = params.get("HasAcceptedTerms") == terms_version
    completed_training = params.get("CompletedTrainingVersion") == training_version

    panda_signature = params.get("PandaFirmware")
    fw_version_match = (panda_signature is None) or (panda_signature == FW_SIGNATURE)   # don't show alert is no panda is connected (None)

    should_start = ignition

    # with 2% left, we killall, otherwise the phone will take a long time to boot
    should_start = should_start and msg.thermal.freeSpace > 0.02

    # confirm we have completed training and aren't uninstalling
    should_start = should_start and accepted_terms and completed_training and (not do_uninstall)

    # check for firmware mismatch
    should_start = should_start and fw_version_match

    # check if system time is valid
    should_start = should_start and time_valid

    # don't start while taking snapshot
    if not should_start_prev:
      is_viewing_driver = params.get("IsDriverViewEnabled") == b"1"
      is_taking_snapshot = params.get("IsTakingSnapshot") == b"1"
      should_start = should_start and (not is_taking_snapshot) and (not is_viewing_driver)

    if fw_version_match and not fw_version_match_prev:
      set_offroad_alert("Offroad_PandaFirmwareMismatch", False)
    if not fw_version_match and fw_version_match_prev:
      set_offroad_alert("Offroad_PandaFirmwareMismatch", True)

    # if any CPU gets above 107 or the battery gets above 63, kill all processes
    # controls will warn with CPU above 95 or battery above 60
    if thermal_status >= ThermalStatus.danger:
      should_start = False
      if thermal_status_prev < ThermalStatus.danger:
        set_offroad_alert("Offroad_TemperatureTooHigh", True)
    else:
      if thermal_status_prev >= ThermalStatus.danger:
        set_offroad_alert("Offroad_TemperatureTooHigh", False)

    if should_start:
      if not should_start_prev:
        params.delete("IsOffroad")

      off_ts = None
      if started_ts is None:
        started_ts = sec_since_boot()
        started_seen = True
        os.system('echo performance > /sys/class/devfreq/soc:qcom,cpubw/governor')
    else:
      if should_start_prev or (count == 0):
        put_nonblocking("IsOffroad", "1")

      started_ts = None
      if off_ts is None:
        off_ts = sec_since_boot()
        os.system('echo powersave > /sys/class/devfreq/soc:qcom,cpubw/governor')

    # Offroad power monitoring
    pm.calculate(health)
    msg.thermal.offroadPowerUsage = pm.get_power_used()
    msg.thermal.carBatteryCapacity = max(0, pm.get_car_battery_capacity())

    # Check if we need to disable charging (handled by boardd)
    msg.thermal.chargingDisabled = pm.should_disable_charging(health, off_ts)

    # Check if we need to shut down
    if pm.should_shutdown(health, off_ts, started_seen, LEON):
      cloudlog.info(f"shutting device down, offroad since {off_ts}")
      # TODO: add function for blocking cloudlog instead of sleep
      time.sleep(10)
      os.system('LD_LIBRARY_PATH="" svc power shutdown')

    msg.thermal.chargingError = current_filter.x > 0. and msg.thermal.batteryPercent < 90  # if current is positive, then battery is being discharged
    msg.thermal.started = started_ts is not None
    msg.thermal.startedTs = int(1e9*(started_ts or 0))

    msg.thermal.thermalStatus = thermal_status
    thermal_sock.send(msg.to_bytes())

    if usb_power_prev and not usb_power:
      set_offroad_alert("Offroad_ChargeDisabled", True)
    elif usb_power and not usb_power_prev:
      set_offroad_alert("Offroad_ChargeDisabled", False)

    thermal_status_prev = thermal_status
    usb_power_prev = usb_power
    fw_version_match_prev = fw_version_match
    should_start_prev = should_start

    # report to server once per minute
    if (count % int(60. / DT_TRML)) == 0:
      cloudlog.event("STATUS_PACKET",
                     count=count,
                     health=(health.to_dict() if health else None),
                     location=(location.to_dict() if location else None),
                     thermal=msg.to_dict())

    count += 1


def main():
  thermald_thread()


if __name__ == "__main__":
  main()
