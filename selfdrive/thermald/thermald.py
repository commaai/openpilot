#!/usr/bin/env python3
import datetime
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil
from smbus2 import SMBus

import cereal.messaging as messaging
from cereal import log
from common.filter_simple import FirstOrderFilter
from common.numpy_fast import clip, interp
from common.params import Params, ParamKeyType
from common.realtime import DT_TRML, sec_since_boot
# from common.dict_helpers import strip_deprecated_keys
from selfdrive.controls.lib.alertmanager import set_offroad_alert
from selfdrive.hardware import EON, TICI, HARDWARE
from selfdrive.loggerd.config import get_available_percent
from selfdrive.pandad import get_expected_signature
from selfdrive.swaglog import cloudlog
from selfdrive.thermald.power_monitoring import PowerMonitoring, MAX_TIME_OFFROAD_S
from selfdrive.version import get_git_branch, terms_version, training_version

FW_SIGNATURE = get_expected_signature()

ThermalStatus = log.DeviceState.ThermalStatus
NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength
CURRENT_TAU = 15.   # 15s time constant
CPU_TEMP_TAU = 5.   # 5s time constant
DAYS_NO_CONNECTIVITY_MAX = 7  # do not allow to engage after a week without internet
DAYS_NO_CONNECTIVITY_PROMPT = 4  # send an offroad prompt after 4 days with no internet
DISCONNECT_TIMEOUT = 5.  # wait 5 seconds before going offroad after disconnect so you get an alert

prev_offroad_states: Dict[str, Tuple[bool, Optional[str]]] = {}

LEON = False
last_eon_fan_val = None

params = Params()
from common.dp_time import LAST_MODIFIED_THERMALD
from common.dp_common import get_last_modified, param_get_if_updated

def read_tz(x):
  if x is None:
    return 0

  try:
    with open(f"/sys/devices/virtual/thermal/thermal_zone{x}/temp") as f:
      return int(f.read())
  except FileNotFoundError:
    return 0


def read_thermal(thermal_config):
  dat = messaging.new_message('deviceState')
  dat.deviceState.cpuTempC = [read_tz(z) / thermal_config.cpu[1] for z in thermal_config.cpu[0]]
  dat.deviceState.gpuTempC = [read_tz(z) / thermal_config.gpu[1] for z in thermal_config.gpu[0]]
  dat.deviceState.memoryTempC = read_tz(thermal_config.mem[0]) / thermal_config.mem[1]
  dat.deviceState.ambientTempC = read_tz(thermal_config.ambient[0]) / thermal_config.ambient[1]
  dat.deviceState.batteryTempC = read_tz(thermal_config.bat[0]) / thermal_config.bat[1]
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
_BAT_TEMP_THRESHOLD = 45.


def handle_fan_eon(dp_fan_mode, max_cpu_temp, bat_temp, fan_speed, ignition):
  _fan_speed = _FAN_SPEEDS
  _bat_temp_threshold = _BAT_TEMP_THRESHOLD
  if dp_fan_mode == 2:
    _fan_speed = [0, 65535, 65535, 65535]
    _bat_temp_threshold = 15.
  elif dp_fan_mode == 1:
    _fan_speed = [0, 16384, 16384, 32768]

  new_speed_h = next(speed for speed, temp_h in zip(_fan_speed, _TEMP_THRS_H) if temp_h > max_cpu_temp)
  new_speed_l = next(speed for speed, temp_l in zip(_fan_speed, _TEMP_THRS_L) if temp_l > max_cpu_temp)

  if new_speed_h > fan_speed:
    # update speed if using the high thresholds results in fan speed increment
    fan_speed = new_speed_h
  elif new_speed_l < fan_speed:
    # update speed if using the low thresholds results in fan speed decrement
    fan_speed = new_speed_l

  if bat_temp < _bat_temp_threshold:
    # no max fan speed unless battery is hot
    fan_speed = min(fan_speed, _fan_speed[-2])

  set_eon_fan(fan_speed // 16384)

  return fan_speed


def handle_fan_uno(dp_fan_mode, max_cpu_temp, bat_temp, fan_speed, ignition):
  if dp_fan_mode == 2:
    new_speed = 80
  elif dp_fan_mode == 1:
    new_speed = int(interp(max_cpu_temp, [65.0, 80.0, 90.0], [0, 20, 60]))
  else:
    new_speed = int(interp(max_cpu_temp, [40.0, 80.0], [0, 80]))

  if not ignition:
    new_speed = min(10 if dp_fan_mode == 2 else 30, new_speed)

  return new_speed


def set_offroad_alert_if_changed(offroad_alert: str, show_alert: bool, extra_text: Optional[str]=None):
  if prev_offroad_states.get(offroad_alert, None) == (show_alert, extra_text):
    return
  prev_offroad_states[offroad_alert] = (show_alert, extra_text)
  set_offroad_alert(offroad_alert, show_alert, extra_text)


def thermald_thread():

  pm = messaging.PubMaster(['deviceState'])

  pandaState_timeout = int(1000 * 2.5 * DT_TRML)  # 2.5x the expected pandaState frequency
  pandaState_sock = messaging.sub_sock('pandaState', timeout=pandaState_timeout)
  location_sock = messaging.sub_sock('gpsLocationExternal')
  managerState_sock = messaging.sub_sock('managerState', conflate=True)

  fan_speed = 0
  count = 0

  startup_conditions = {
    "ignition": False,
  }
  startup_conditions_prev = startup_conditions.copy()

  off_ts = None
  started_ts = None
  started_seen = False
  thermal_status = ThermalStatus.green
  usb_power = True
  current_branch = get_git_branch()

  network_type = NetworkType.none
  network_strength = NetworkStrength.unknown
  network_info = None

  current_filter = FirstOrderFilter(0., CURRENT_TAU, DT_TRML)
  cpu_temp_filter = FirstOrderFilter(0., CPU_TEMP_TAU, DT_TRML)
  pandaState_prev = None
  should_start_prev = False
  handle_fan = None
  is_uno = False
  ui_running_prev = False

  power_monitor = PowerMonitoring()
  no_panda_cnt = 0

  thermal_config = HARDWARE.get_thermal_config()

  if params.get_bool("IsOnroad"):
    cloudlog.event("onroad flag not cleared")

  # dp
  dp_no_batt = params.get_bool("dp_no_batt")
  dp_temp_monitor = True
  dp_last_modified_temp_monitor = None

  dp_debug = False
  dp_last_modified_debug = None

  dp_auto_shutdown = False
  dp_last_modified_auto_shutdown = None
  dp_auto_shutdown_last = False

  dp_auto_shutdown_in = 90
  dp_last_modified_auto_shutdown_in = None
  dp_auto_shutdown_in_last = 90

  dp_fan_mode = 0
  dp_fan_mode_last = None

  modified = None
  last_modified = None
  last_modified_check = None

  # CPR3 logging
  if EON:
    base_path = "/sys/kernel/debug/cpr3-regulator/"
    cpr_files = [p for p in Path(base_path).glob("**/*") if p.is_file()]
    cpr_files = ["/sys/kernel/debug/regulator/pm8994_s11/voltage"] + cpr_files
    cpr_data = {}
    for cf in cpr_files:
      with open(cf, "r") as f:
        try:
          cpr_data[str(cf)] = f.read().strip()
        except Exception:
          pass
    cloudlog.event("CPR", data=cpr_data)

  while 1:

    # dp - load temp monitor conf
    last_modified_check, modified = get_last_modified(LAST_MODIFIED_THERMALD, last_modified_check, modified)
    if last_modified != modified:
      dp_temp_monitor, dp_last_modified_temp_monitor = param_get_if_updated("dp_temp_monitor", "bool", dp_temp_monitor, dp_last_modified_temp_monitor)
      dp_auto_shutdown, dp_last_modified_auto_shutdown = param_get_if_updated("dp_auto_shutdown", "bool", dp_auto_shutdown, dp_last_modified_auto_shutdown)
      dp_auto_shutdown_in, dp_last_modified_auto_shutdown_in = param_get_if_updated("dp_auto_shutdown_in", "int", dp_auto_shutdown_in, dp_last_modified_auto_shutdown_in)
      dp_fan_mode, dp_fan_mode_last = param_get_if_updated("dp_fan_mode", "int", dp_fan_mode, dp_fan_mode_last)
      dp_debug, dp_last_modified_debug = param_get_if_updated("dp_debug", "int", dp_debug, dp_last_modified_debug)
      last_modified = modified

    pandaState = messaging.recv_sock(pandaState_sock, wait=True)
    msg = read_thermal(thermal_config)

    if pandaState is not None:
      usb_power = pandaState.pandaState.usbPowerMode != log.PandaState.UsbPowerMode.client

      # If we lose connection to the panda, wait 5 seconds before going offroad
      if pandaState.pandaState.pandaType == log.PandaState.PandaType.unknown:
        no_panda_cnt += 1
        if no_panda_cnt > DISCONNECT_TIMEOUT / DT_TRML:
          if startup_conditions["ignition"]:
            cloudlog.error("Lost panda connection while onroad")
          startup_conditions["ignition"] = False
      else:
        no_panda_cnt = 0
        startup_conditions["ignition"] = pandaState.pandaState.ignitionLine or pandaState.pandaState.ignitionCan

      startup_conditions["hardware_supported"] = pandaState.pandaState.pandaType not in [log.PandaState.PandaType.whitePanda,
                                                                                         log.PandaState.PandaType.greyPanda]
      set_offroad_alert_if_changed("Offroad_HardwareUnsupported", not startup_conditions["hardware_supported"])

      # Setup fan handler on first connect to panda
      if handle_fan is None and pandaState.pandaState.pandaType != log.PandaState.PandaType.unknown:
        is_uno = pandaState.pandaState.pandaType == log.PandaState.PandaType.uno

        if (not EON) or is_uno:
          cloudlog.info("Setting up UNO fan handler")
          handle_fan = handle_fan_uno
        else:
          cloudlog.info("Setting up EON fan handler")
          setup_eon_fan()
          handle_fan = handle_fan_eon

      # Handle disconnect
      if pandaState_prev is not None:
        if pandaState.pandaState.pandaType == log.PandaState.PandaType.unknown and \
          pandaState_prev.pandaState.pandaType != log.PandaState.PandaType.unknown:
          params.clear_all(ParamKeyType.CLEAR_ON_PANDA_DISCONNECT)
      pandaState_prev = pandaState

    # get_network_type is an expensive call. update every 10s
    if (count % int(10. / DT_TRML)) == 0:
      try:
        network_type = HARDWARE.get_network_type()
        network_strength = HARDWARE.get_network_strength(network_type)
        network_info = HARDWARE.get_network_info()  # pylint: disable=assignment-from-none
      except Exception:
        cloudlog.exception("Error getting network status")

    msg.deviceState.freeSpacePercent = get_available_percent(default=100.0)
    msg.deviceState.memoryUsagePercent = int(round(psutil.virtual_memory().percent))
    msg.deviceState.cpuUsagePercent = int(round(psutil.cpu_percent()))
    msg.deviceState.networkType = network_type
    msg.deviceState.networkStrength = network_strength
    if network_info is not None:
      msg.deviceState.networkInfo = network_info

    msg.deviceState.batteryPercent = HARDWARE.get_battery_capacity()
    msg.deviceState.batteryStatus = HARDWARE.get_battery_status()
    msg.deviceState.batteryCurrent = HARDWARE.get_battery_current()
    msg.deviceState.batteryVoltage = HARDWARE.get_battery_voltage()
    msg.deviceState.usbOnline = HARDWARE.get_usb_present()

    # Fake battery levels on uno for frame
    if (not EON) or is_uno or dp_no_batt:
      msg.deviceState.batteryPercent = 100
      msg.deviceState.batteryStatus = "Charging"
      msg.deviceState.batteryTempC = 0

    current_filter.update(msg.deviceState.batteryCurrent / 1e6)

    # TODO: add car battery voltage check
    max_cpu_temp = cpu_temp_filter.update(max(msg.deviceState.cpuTempC))
    max_comp_temp = max(max_cpu_temp, msg.deviceState.memoryTempC, max(msg.deviceState.gpuTempC))
    bat_temp = msg.deviceState.batteryTempC

    if handle_fan is not None:
      fan_speed = handle_fan(dp_fan_mode, max_cpu_temp, bat_temp, fan_speed, startup_conditions["ignition"])
      msg.deviceState.fanSpeedPercentDesired = fan_speed

    # If device is offroad we want to cool down before going onroad
    # since going onroad increases load and can make temps go over 107
    # We only do this if there is a relay that prevents the car from faulting
    is_offroad_for_5_min = (started_ts is None) and ((not started_seen) or (off_ts is None) or (sec_since_boot() - off_ts > 60 * 5))
    if max_cpu_temp > 107. or bat_temp >= 63. or (is_offroad_for_5_min and max_cpu_temp > 70.0):
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
      thermal_status = ThermalStatus.green  # default to good condition

    if not dp_temp_monitor and thermal_status in [ThermalStatus.red, ThermalStatus.danger]:
      thermal_status = ThermalStatus.yellow
    # **** starting logic ****

    # Check for last update time and display alerts if needed
    # now = datetime.datetime.utcnow()
    #
    # # show invalid date/time alert
    # startup_conditions["time_valid"] = (now.year > 2020) or (now.year == 2020 and now.month >= 10)
    # set_offroad_alert_if_changed("Offroad_InvalidTime", (not startup_conditions["time_valid"]))
    #
    # # Show update prompt
    # try:
    #   last_update = datetime.datetime.fromisoformat(params.get("LastUpdateTime", encoding='utf8'))
    # except (TypeError, ValueError):
    #   last_update = now
    # dt = now - last_update
    #
    # update_failed_count = params.get("UpdateFailedCount")
    # update_failed_count = 0 if update_failed_count is None else int(update_failed_count)
    # last_update_exception = params.get("LastUpdateException", encoding='utf8')
    #
    # if update_failed_count > 15 and last_update_exception is not None:
    #   if current_branch in ["release2", "dashcam"]:
    #     extra_text = "Ensure the software is correctly installed"
    #   else:
    #     extra_text = last_update_exception
    #
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeeded", False)
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeededPrompt", False)
    #   set_offroad_alert_if_changed("Offroad_UpdateFailed", True, extra_text=extra_text)
    # elif dt.days > DAYS_NO_CONNECTIVITY_MAX and update_failed_count > 1:
    #   set_offroad_alert_if_changed("Offroad_UpdateFailed", False)
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeededPrompt", False)
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeeded", True)
    # elif dt.days > DAYS_NO_CONNECTIVITY_PROMPT:
    #   remaining_time = str(max(DAYS_NO_CONNECTIVITY_MAX - dt.days, 0))
    #   set_offroad_alert_if_changed("Offroad_UpdateFailed", False)
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeeded", False)
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeededPrompt", True, extra_text=f"{remaining_time} days.")
    # else:
    #   set_offroad_alert_if_changed("Offroad_UpdateFailed", False)
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeeded", False)
    #   set_offroad_alert_if_changed("Offroad_ConnectivityNeededPrompt", False)

    startup_conditions["up_to_date"] = params.get("Offroad_ConnectivityNeeded") is None or params.get_bool("DisableUpdates")
    startup_conditions["not_uninstalling"] = not params.get_bool("DoUninstall")
    startup_conditions["accepted_terms"] = params.get("HasAcceptedTerms") == terms_version

    panda_signature = params.get("PandaFirmware")
    startup_conditions["fw_version_match"] = (panda_signature is None) or (panda_signature == FW_SIGNATURE)   # don't show alert is no panda is connected (None)
    set_offroad_alert_if_changed("Offroad_PandaFirmwareMismatch", (not startup_conditions["fw_version_match"]))

    # with 2% left, we killall, otherwise the phone will take a long time to boot
    startup_conditions["free_space"] = msg.deviceState.freeSpacePercent > 2
    startup_conditions["completed_training"] = params.get("CompletedTrainingVersion") == training_version or \
                                               (current_branch in ['dashcam', 'dashcam-staging'])
    startup_conditions["not_driver_view"] = not params.get_bool("IsDriverViewEnabled")
    startup_conditions["not_taking_snapshot"] = not params.get_bool("IsTakingSnapshot")
    # if any CPU gets above 107 or the battery gets above 63, kill all processes
    # controls will warn with CPU above 95 or battery above 60
    startup_conditions["device_temp_good"] = thermal_status < ThermalStatus.danger
    set_offroad_alert_if_changed("Offroad_TemperatureTooHigh", (not startup_conditions["device_temp_good"]))

    if TICI:
      set_offroad_alert_if_changed("Offroad_NvmeMissing", (not Path("/data/media").is_mount()))

    # Handle offroad/onroad transition
    should_start = all(startup_conditions.values())
    if dp_debug:
      should_start = True
    if should_start != should_start_prev or (count == 0):
      params.put_bool("IsOnroad", should_start)
      params.put_bool("IsOffroad", not should_start)
      HARDWARE.set_power_save(not should_start)
      if TICI and not params.get_bool("EnableLteOnroad"):
        fxn = "off" if should_start else "on"
        os.system(f"nmcli radio wwan {fxn}")

    if should_start:
      off_ts = None
      if started_ts is None:
        started_ts = sec_since_boot()
        started_seen = True
    else:
      if startup_conditions["ignition"] and (startup_conditions != startup_conditions_prev):
        cloudlog.event("Startup blocked", startup_conditions=startup_conditions)

      started_ts = None
      if off_ts is None:
        off_ts = sec_since_boot()

    # Offroad power monitoring
    if not dp_no_batt:
      power_monitor.calculate(pandaState)
      msg.deviceState.offroadPowerUsageUwh = power_monitor.get_power_used()
      msg.deviceState.carBatteryCapacityUwh = max(0, power_monitor.get_car_battery_capacity())

    # Check if we need to disable charging (handled by boardd)
    msg.deviceState.chargingDisabled = power_monitor.should_disable_charging(pandaState, off_ts)

    # Check if we need to shut down
    if power_monitor.should_shutdown(pandaState, off_ts, started_seen, LEON):
      cloudlog.info(f"shutting device down, offroad since {off_ts}")
      # TODO: add function for blocking cloudlog instead of sleep
      time.sleep(10)
      HARDWARE.shutdown()

    # dp - auto shutdown
    # reset off_ts if we change auto shutdown related params
    if off_ts is not None:
      if dp_auto_shutdown:
        shutdown_sec = dp_auto_shutdown_in * 60
        sec_now = sec_since_boot() - off_ts
        if (shutdown_sec - 5) < sec_now:
          msg.deviceState.chargingDisabled = True
        if shutdown_sec < sec_now:
          time.sleep(10)
          HARDWARE.shutdown()

      if dp_auto_shutdown_in_last != dp_auto_shutdown_in or dp_auto_shutdown_last != dp_auto_shutdown:
        off_ts = sec_since_boot()
      dp_auto_shutdown_last = dp_auto_shutdown
      dp_auto_shutdown_in_last = dp_auto_shutdown_in

    # If UI has crashed, set the brightness to reasonable non-zero value
    manager_state = messaging.recv_one_or_none(managerState_sock)
    if manager_state is not None:
      ui_running = "ui" in (p.name for p in manager_state.managerState.processes if p.running)
      if ui_running_prev and not ui_running:
        HARDWARE.set_screen_brightness(20)
      ui_running_prev = ui_running

    msg.deviceState.chargingError = current_filter.x > 0. and msg.deviceState.batteryPercent < 90  # if current is positive, then battery is being discharged
    msg.deviceState.started = started_ts is not None
    msg.deviceState.startedMonoTime = int(1e9*(started_ts or 0))

    last_ping = params.get("LastAthenaPingTime")
    if last_ping is not None:
      msg.deviceState.lastAthenaPingTime = int(last_ping)

    msg.deviceState.thermalStatus = thermal_status
    pm.send("deviceState", msg)

    if EON and not is_uno:
      set_offroad_alert_if_changed("Offroad_ChargeDisabled", (not usb_power))

    should_start_prev = should_start
    startup_conditions_prev = startup_conditions.copy()

    # report to server once every 10 minutes
    # if (count % int(600. / DT_TRML)) == 0:
    #   if EON and started_ts is None and msg.deviceState.memoryUsagePercent > 40:
    #     cloudlog.event("High offroad memory usage", mem=msg.deviceState.memoryUsagePercent)
    #
    #   location = messaging.recv_sock(location_sock)
    #   cloudlog.event("STATUS_PACKET",
    #                  count=count,
    #                  pandaState=(strip_deprecated_keys(pandaState.to_dict()) if pandaState else None),
    #                  location=(strip_deprecated_keys(location.gpsLocationExternal.to_dict()) if location else None),
    #                  deviceState=strip_deprecated_keys(msg.to_dict()))

    count += 1


def main():
  thermald_thread()


if __name__ == "__main__":
  main()
