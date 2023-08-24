#!/usr/bin/env python3
import datetime
import os
import json
import queue
import threading
import time
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil

import cereal.messaging as messaging
from cereal import log
from openpilot.common.dict_helpers import strip_deprecated_keys
from openpilot.common.time import MIN_DATE
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_TRML
from openpilot.selfdrive.controls.lib.alertmanager import set_offroad_alert
from openpilot.system.hardware import HARDWARE, TICI, AGNOS
from openpilot.system.loggerd.config import get_available_percent
from openpilot.selfdrive.statsd import statlog
from openpilot.system.swaglog import cloudlog
from openpilot.selfdrive.thermald.power_monitoring import PowerMonitoring
from openpilot.selfdrive.thermald.fan_controller import TiciFanController
from openpilot.system.version import terms_version, training_version

ThermalStatus = log.DeviceState.ThermalStatus
NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength
CURRENT_TAU = 15.   # 15s time constant
TEMP_TAU = 5.   # 5s time constant
DISCONNECT_TIMEOUT = 5.  # wait 5 seconds before going offroad after disconnect so you get an alert
PANDA_STATES_TIMEOUT = int(1000 * 1.5 * DT_TRML)  # 1.5x the expected pandaState frequency

ThermalBand = namedtuple("ThermalBand", ['min_temp', 'max_temp'])
HardwareState = namedtuple("HardwareState", ['network_type', 'network_info', 'network_strength', 'network_stats',
                                             'network_metered', 'nvme_temps', 'modem_temps'])

# List of thermal bands. We will stay within this region as long as we are within the bounds.
# When exiting the bounds, we'll jump to the lower or higher band. Bands are ordered in the dict.
THERMAL_BANDS = OrderedDict({
  ThermalStatus.green: ThermalBand(None, 80.0),
  ThermalStatus.yellow: ThermalBand(75.0, 96.0),
  ThermalStatus.red: ThermalBand(88.0, 107.),
  ThermalStatus.danger: ThermalBand(94.0, None),
})

# Override to highest thermal band when offroad and above this temp
OFFROAD_DANGER_TEMP = 75

prev_offroad_states: Dict[str, Tuple[bool, Optional[str]]] = {}

tz_by_type: Optional[Dict[str, int]] = None
def populate_tz_by_type():
  global tz_by_type
  tz_by_type = {}
  for n in os.listdir("/sys/devices/virtual/thermal"):
    if not n.startswith("thermal_zone"):
      continue
    with open(os.path.join("/sys/devices/virtual/thermal", n, "type")) as f:
      tz_by_type[f.read().strip()] = int(n.removeprefix("thermal_zone"))

def read_tz(x):
  if x is None:
    return 0

  if isinstance(x, str):
    if tz_by_type is None:
      populate_tz_by_type()
    x = tz_by_type[x]

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
  dat.deviceState.pmicTempC = [read_tz(z) / thermal_config.pmic[1] for z in thermal_config.pmic[0]]
  return dat


def set_offroad_alert_if_changed(offroad_alert: str, show_alert: bool, extra_text: Optional[str]=None):
  if prev_offroad_states.get(offroad_alert, None) == (show_alert, extra_text):
    return
  prev_offroad_states[offroad_alert] = (show_alert, extra_text)
  set_offroad_alert(offroad_alert, show_alert, extra_text)


def hw_state_thread(end_event, hw_queue):
  """Handles non critical hardware state, and sends over queue"""
  count = 0
  prev_hw_state = None

  modem_version = None
  modem_nv = None
  modem_configured = False
  modem_restarted = False
  modem_missing_count = 0

  while not end_event.is_set():
    # these are expensive calls. update every 10s
    if (count % int(10. / DT_TRML)) == 0:
      try:
        network_type = HARDWARE.get_network_type()
        modem_temps = HARDWARE.get_modem_temperatures()
        if len(modem_temps) == 0 and prev_hw_state is not None:
          modem_temps = prev_hw_state.modem_temps

        # Log modem version once
        if AGNOS and ((modem_version is None) or (modem_nv is None)):
          modem_version = HARDWARE.get_modem_version()  # pylint: disable=assignment-from-none
          modem_nv = HARDWARE.get_modem_nv()  # pylint: disable=assignment-from-none

          if (modem_version is not None) and (modem_nv is not None):
            cloudlog.event("modem version", version=modem_version, nv=modem_nv)
          else:
            if not modem_restarted:
              # TODO: we may be able to remove this with a MM update
              # ModemManager's probing on startup can fail
              # rarely, restart the service to probe again.
              modem_missing_count += 1
              if modem_missing_count > 3:
                modem_restarted = True
                cloudlog.event("restarting ModemManager")
                os.system("sudo systemctl restart --no-block ModemManager")

        tx, rx = HARDWARE.get_modem_data_usage()

        hw_state = HardwareState(
          network_type=network_type,
          network_info=HARDWARE.get_network_info(),
          network_strength=HARDWARE.get_network_strength(network_type),
          network_stats={'wwanTx': tx, 'wwanRx': rx},
          network_metered=HARDWARE.get_network_metered(network_type),
          nvme_temps=HARDWARE.get_nvme_temperatures(),
          modem_temps=modem_temps,
        )

        try:
          hw_queue.put_nowait(hw_state)
        except queue.Full:
          pass

        # TODO: remove this once the config is in AGNOS
        if not modem_configured and len(HARDWARE.get_sim_info().get('sim_id', '')) > 0:
          cloudlog.warning("configuring modem")
          HARDWARE.configure_modem()
          modem_configured = True

        prev_hw_state = hw_state
      except Exception:
        cloudlog.exception("Error getting hardware state")

    count += 1
    time.sleep(DT_TRML)


def thermald_thread(end_event, hw_queue) -> None:
  pm = messaging.PubMaster(['deviceState'])
  sm = messaging.SubMaster(["peripheralState", "gpsLocationExternal", "controlsState", "pandaStates"], poll=["pandaStates"])

  count = 0

  onroad_conditions: Dict[str, bool] = {
    "ignition": False,
  }
  startup_conditions: Dict[str, bool] = {}
  startup_conditions_prev: Dict[str, bool] = {}

  off_ts: Optional[float] = None
  started_ts: Optional[float] = None
  started_seen = False
  startup_blocked_ts: Optional[float] = None
  thermal_status = ThermalStatus.yellow

  last_hw_state = HardwareState(
    network_type=NetworkType.none,
    network_info=None,
    network_metered=False,
    network_strength=NetworkStrength.unknown,
    network_stats={'wwanTx': -1, 'wwanRx': -1},
    nvme_temps=[],
    modem_temps=[],
  )

  all_temp_filter = FirstOrderFilter(0., TEMP_TAU, DT_TRML, initialized=False)
  offroad_temp_filter = FirstOrderFilter(0., TEMP_TAU, DT_TRML, initialized=False)
  should_start_prev = False
  in_car = False
  engaged_prev = False

  params = Params()
  power_monitor = PowerMonitoring()

  HARDWARE.initialize_hardware()
  thermal_config = HARDWARE.get_thermal_config()

  fan_controller = None

  while not end_event.is_set():
    sm.update(PANDA_STATES_TIMEOUT)

    pandaStates = sm['pandaStates']
    peripheralState = sm['peripheralState']
    peripheral_panda_present = peripheralState.pandaType != log.PandaState.PandaType.unknown

    msg = read_thermal(thermal_config)

    if sm.updated['pandaStates'] and len(pandaStates) > 0:

      # Set ignition based on any panda connected
      onroad_conditions["ignition"] = any(ps.ignitionLine or ps.ignitionCan for ps in pandaStates if ps.pandaType != log.PandaState.PandaType.unknown)

      pandaState = pandaStates[0]

      in_car = pandaState.harnessStatus != log.PandaState.HarnessStatus.notConnected

      # Setup fan handler on first connect to panda
      if fan_controller is None and peripheral_panda_present:
        if TICI:
          fan_controller = TiciFanController()

    elif (time.monotonic() - sm.rcv_time['pandaStates']) > DISCONNECT_TIMEOUT:
      if onroad_conditions["ignition"]:
        onroad_conditions["ignition"] = False
        cloudlog.error("panda timed out onroad")

    try:
      last_hw_state = hw_queue.get_nowait()
    except queue.Empty:
      pass

    msg.deviceState.freeSpacePercent = get_available_percent(default=100.0)
    msg.deviceState.memoryUsagePercent = int(round(psutil.virtual_memory().percent))
    msg.deviceState.cpuUsagePercent = [int(round(n)) for n in psutil.cpu_percent(percpu=True)]
    msg.deviceState.gpuUsagePercent = int(round(HARDWARE.get_gpu_usage_percent()))

    msg.deviceState.networkType = last_hw_state.network_type
    msg.deviceState.networkMetered = last_hw_state.network_metered
    msg.deviceState.networkStrength = last_hw_state.network_strength
    msg.deviceState.networkStats = last_hw_state.network_stats
    if last_hw_state.network_info is not None:
      msg.deviceState.networkInfo = last_hw_state.network_info

    msg.deviceState.nvmeTempC = last_hw_state.nvme_temps
    msg.deviceState.modemTempC = last_hw_state.modem_temps

    msg.deviceState.screenBrightnessPercent = HARDWARE.get_screen_brightness()

    # this subset is only used for offroad
    temp_sources = [
      msg.deviceState.memoryTempC,
      max(msg.deviceState.cpuTempC),
      max(msg.deviceState.gpuTempC),
    ]
    offroad_comp_temp = offroad_temp_filter.update(max(temp_sources))

    # this drives the thermal status while onroad
    temp_sources.append(max(msg.deviceState.pmicTempC))
    all_comp_temp = all_temp_filter.update(max(temp_sources))
    msg.deviceState.maxTempC = all_comp_temp

    if fan_controller is not None:
      msg.deviceState.fanSpeedPercentDesired = fan_controller.update(all_comp_temp, onroad_conditions["ignition"])

    is_offroad_for_5_min = (started_ts is None) and ((not started_seen) or (off_ts is None) or (time.monotonic() - off_ts > 60 * 5))
    if is_offroad_for_5_min and offroad_comp_temp > OFFROAD_DANGER_TEMP:
      # if device is offroad and already hot without the extra onroad load,
      # we want to cool down first before increasing load
      thermal_status = ThermalStatus.danger
    else:
      current_band = THERMAL_BANDS[thermal_status]
      band_idx = list(THERMAL_BANDS.keys()).index(thermal_status)
      if current_band.min_temp is not None and all_comp_temp < current_band.min_temp:
        thermal_status = list(THERMAL_BANDS.keys())[band_idx - 1]
      elif current_band.max_temp is not None and all_comp_temp > current_band.max_temp:
        thermal_status = list(THERMAL_BANDS.keys())[band_idx + 1]

    # **** starting logic ****

    # Ensure date/time are valid
    now = datetime.datetime.utcnow()
    startup_conditions["time_valid"] = now > MIN_DATE
    set_offroad_alert_if_changed("Offroad_InvalidTime", (not startup_conditions["time_valid"]) and peripheral_panda_present)

    startup_conditions["up_to_date"] = params.get("Offroad_ConnectivityNeeded") is None or params.get_bool("DisableUpdates") or params.get_bool("SnoozeUpdate")
    startup_conditions["not_uninstalling"] = not params.get_bool("DoUninstall")
    startup_conditions["accepted_terms"] = params.get("HasAcceptedTerms") == terms_version

    # with 2% left, we killall, otherwise the phone will take a long time to boot
    startup_conditions["free_space"] = msg.deviceState.freeSpacePercent > 2
    startup_conditions["completed_training"] = params.get("CompletedTrainingVersion") == training_version or \
                                               params.get_bool("Passive")
    startup_conditions["not_driver_view"] = not params.get_bool("IsDriverViewEnabled")
    startup_conditions["not_taking_snapshot"] = not params.get_bool("IsTakingSnapshot")

    # must be at an engageable thermal band to go onroad
    startup_conditions["device_temp_engageable"] = thermal_status < ThermalStatus.red

    # if the temperature enters the danger zone, go offroad to cool down
    onroad_conditions["device_temp_good"] = thermal_status < ThermalStatus.danger
    extra_text = f"{offroad_comp_temp:.1f}C"
    show_alert = (not onroad_conditions["device_temp_good"] or not startup_conditions["device_temp_engageable"]) and onroad_conditions["ignition"]
    set_offroad_alert_if_changed("Offroad_TemperatureTooHigh", show_alert, extra_text=extra_text)

    # TODO: this should move to TICI.initialize_hardware, but we currently can't import params there
    if TICI:
      if not os.path.isfile("/persist/comma/living-in-the-moment"):
        if not Path("/data/media").is_mount():
          set_offroad_alert_if_changed("Offroad_StorageMissing", True)
        else:
          # check for bad NVMe
          try:
            with open("/sys/block/nvme0n1/device/model") as f:
              model = f.read().strip()
            if not model.startswith("Samsung SSD 980") and params.get("Offroad_BadNvme") is None:
              set_offroad_alert_if_changed("Offroad_BadNvme", True)
              cloudlog.event("Unsupported NVMe", model=model, error=True)
          except Exception:
            pass

    # Handle offroad/onroad transition
    should_start = all(onroad_conditions.values())
    if started_ts is None:
      should_start = should_start and all(startup_conditions.values())

    if should_start != should_start_prev or (count == 0):
      params.put_bool("IsEngaged", False)
      engaged_prev = False
      HARDWARE.set_power_save(not should_start)

    if sm.updated['controlsState']:
      engaged = sm['controlsState'].enabled
      if engaged != engaged_prev:
        params.put_bool("IsEngaged", engaged)
        engaged_prev = engaged

      try:
        with open('/dev/kmsg', 'w') as kmsg:
          kmsg.write(f"<3>[thermald] engaged: {engaged}\n")
      except Exception:
        pass

    if should_start:
      off_ts = None
      if started_ts is None:
        started_ts = time.monotonic()
        started_seen = True
        if startup_blocked_ts is not None:
          cloudlog.event("Startup after block", block_duration=(time.monotonic() - startup_blocked_ts),
                         startup_conditions=startup_conditions, onroad_conditions=onroad_conditions,
                         startup_conditions_prev=startup_conditions_prev, error=True)
      startup_blocked_ts = None
    else:
      if onroad_conditions["ignition"] and (startup_conditions != startup_conditions_prev):
        cloudlog.event("Startup blocked", startup_conditions=startup_conditions, onroad_conditions=onroad_conditions, error=True)
        startup_conditions_prev = startup_conditions.copy()
        startup_blocked_ts = time.monotonic()

      started_ts = None
      if off_ts is None:
        off_ts = time.monotonic()

    # Offroad power monitoring
    voltage = None if peripheralState.pandaType == log.PandaState.PandaType.unknown else peripheralState.voltage
    power_monitor.calculate(voltage, onroad_conditions["ignition"])
    msg.deviceState.offroadPowerUsageUwh = power_monitor.get_power_used()
    msg.deviceState.carBatteryCapacityUwh = max(0, power_monitor.get_car_battery_capacity())
    current_power_draw = HARDWARE.get_current_power_draw()
    statlog.sample("power_draw", current_power_draw)
    msg.deviceState.powerDrawW = current_power_draw

    som_power_draw = HARDWARE.get_som_power_draw()
    statlog.sample("som_power_draw", som_power_draw)
    msg.deviceState.somPowerDrawW = som_power_draw

    # Check if we need to shut down
    if power_monitor.should_shutdown(onroad_conditions["ignition"], in_car, off_ts, started_seen):
      cloudlog.warning(f"shutting device down, offroad since {off_ts}")
      params.put_bool("DoShutdown", True)

    msg.deviceState.started = started_ts is not None
    msg.deviceState.startedMonoTime = int(1e9*(started_ts or 0))

    last_ping = params.get("LastAthenaPingTime")
    if last_ping is not None:
      msg.deviceState.lastAthenaPingTime = int(last_ping)

    msg.deviceState.thermalStatus = thermal_status
    pm.send("deviceState", msg)

    # Log to statsd
    statlog.gauge("free_space_percent", msg.deviceState.freeSpacePercent)
    statlog.gauge("gpu_usage_percent", msg.deviceState.gpuUsagePercent)
    statlog.gauge("memory_usage_percent", msg.deviceState.memoryUsagePercent)
    for i, usage in enumerate(msg.deviceState.cpuUsagePercent):
      statlog.gauge(f"cpu{i}_usage_percent", usage)
    for i, temp in enumerate(msg.deviceState.cpuTempC):
      statlog.gauge(f"cpu{i}_temperature", temp)
    for i, temp in enumerate(msg.deviceState.gpuTempC):
      statlog.gauge(f"gpu{i}_temperature", temp)
    statlog.gauge("memory_temperature", msg.deviceState.memoryTempC)
    statlog.gauge("ambient_temperature", msg.deviceState.ambientTempC)
    for i, temp in enumerate(msg.deviceState.pmicTempC):
      statlog.gauge(f"pmic{i}_temperature", temp)
    for i, temp in enumerate(last_hw_state.nvme_temps):
      statlog.gauge(f"nvme_temperature{i}", temp)
    for i, temp in enumerate(last_hw_state.modem_temps):
      statlog.gauge(f"modem_temperature{i}", temp)
    statlog.gauge("fan_speed_percent_desired", msg.deviceState.fanSpeedPercentDesired)
    statlog.gauge("screen_brightness_percent", msg.deviceState.screenBrightnessPercent)

    # report to server once every 10 minutes
    rising_edge_started = should_start and not should_start_prev
    if rising_edge_started or (count % int(600. / DT_TRML)) == 0:
      dat = {
        'count': count,
        'pandaStates': [strip_deprecated_keys(p.to_dict()) for p in pandaStates],
        'peripheralState': strip_deprecated_keys(peripheralState.to_dict()),
        'location': (strip_deprecated_keys(sm["gpsLocationExternal"].to_dict()) if sm.alive["gpsLocationExternal"] else None),
        'deviceState': strip_deprecated_keys(msg.to_dict())
      }
      cloudlog.event("STATUS_PACKET", **dat)

      # save last one before going onroad
      if rising_edge_started:
        try:
          params.put("LastOffroadStatusPacket", json.dumps(dat))
        except Exception:
          cloudlog.exception("failed to save offroad status")

    count += 1
    should_start_prev = should_start


def main():
  hw_queue = queue.Queue(maxsize=1)
  end_event = threading.Event()

  threads = [
    threading.Thread(target=hw_state_thread, args=(end_event, hw_queue)),
    threading.Thread(target=thermald_thread, args=(end_event, hw_queue)),
  ]

  for t in threads:
    t.start()

  try:
    while True:
      time.sleep(1)
      if not all(t.is_alive() for t in threads):
        break
  finally:
    end_event.set()

  for t in threads:
    t.join()


if __name__ == "__main__":
  main()
