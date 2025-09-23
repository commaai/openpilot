#!/usr/bin/env python3
import fcntl
import os
import queue
import struct
import threading
import time
from collections import OrderedDict, namedtuple
from pathlib import Path

import psutil

import cereal.messaging as messaging
from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.common.dict_helpers import strip_deprecated_keys
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_HW
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.system.hardware import HARDWARE, TICI, AGNOS, PC
from openpilot.system.loggerd.config import get_available_percent
from openpilot.system.statsd import statlog
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.power_monitoring import PowerMonitoring
from openpilot.system.hardware.fan_controller import TiciFanController
from openpilot.system.version import terms_version, training_version
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID

ThermalStatus = log.DeviceState.ThermalStatus
NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength
CURRENT_TAU = 15.   # 15s time constant
TEMP_TAU = 5.   # 5s time constant
DISCONNECT_TIMEOUT = 5.  # wait 5 seconds before going offroad after disconnect so you get an alert
PANDA_STATES_TIMEOUT = round(1000 / SERVICE_LIST['pandaStates'].frequency * 1.5)  # 1.5x the expected pandaState frequency
ONROAD_CYCLE_TIME = 1  # seconds to wait offroad after requesting an onroad cycle

ThermalBand = namedtuple("ThermalBand", ['min_temp', 'max_temp'])
HardwareState = namedtuple("HardwareState", ['network_type', 'network_info', 'network_strength', 'network_stats',
                                             'network_metered', 'modem_temps'])

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

prev_offroad_states: dict[str, tuple[bool, str | None]] = {}



def set_offroad_alert_if_changed(offroad_alert: str, show_alert: bool, extra_text: str | None=None):
  if prev_offroad_states.get(offroad_alert, None) == (show_alert, extra_text):
    return
  prev_offroad_states[offroad_alert] = (show_alert, extra_text)
  set_offroad_alert(offroad_alert, show_alert, extra_text)

def touch_thread(end_event):
  count = 0

  pm = messaging.PubMaster(["touch"])

  event_format = "llHHi"
  event_size = struct.calcsize(event_format)
  event_frame = []

  with open("/dev/input/by-path/platform-894000.i2c-event", "rb") as event_file:
    fcntl.fcntl(event_file, fcntl.F_SETFL, os.O_NONBLOCK)
    while not end_event.is_set():
      if (count % int(1. / DT_HW)) == 0:
        event = event_file.read(event_size)
        if event:
          (sec, usec, etype, code, value) = struct.unpack(event_format, event)
          if etype != 0 or code != 0 or value != 0:
            touch = log.Touch.new_message()
            touch.sec = sec
            touch.usec = usec
            touch.type = etype
            touch.code = code
            touch.value = value
            event_frame.append(touch)
          else: # end of frame, push new log
            msg = messaging.new_message('touch', len(event_frame), valid=True)
            msg.touch = event_frame
            pm.send('touch', msg)
            event_frame = []
          continue

      count += 1
      time.sleep(DT_HW)


def hw_state_thread(end_event, hw_queue):
  """Handles non critical hardware state, and sends over queue"""
  count = 0
  prev_hw_state = None

  modem_version = None
  modem_configured = False
  modem_restarted = False
  modem_missing_count = 0

  while not end_event.is_set():
    # these are expensive calls. update every 10s
    if (count % int(10. / DT_HW)) == 0:
      try:
        network_type = HARDWARE.get_network_type()
        modem_temps = HARDWARE.get_modem_temperatures()
        if len(modem_temps) == 0 and prev_hw_state is not None:
          modem_temps = prev_hw_state.modem_temps

        # Log modem version once
        if AGNOS and (modem_version is None):
          modem_version = HARDWARE.get_modem_version()

          if modem_version is not None:
            cloudlog.event("modem version", version=modem_version)
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
          modem_temps=modem_temps,
        )

        try:
          hw_queue.put_nowait(hw_state)
        except queue.Full:
          pass

        if not modem_configured and HARDWARE.get_modem_version() is not None:
          cloudlog.warning("configuring modem")
          HARDWARE.configure_modem()
          modem_configured = True

        prev_hw_state = hw_state
      except Exception:
        cloudlog.exception("Error getting hardware state")

    count += 1
    time.sleep(DT_HW)


def hardware_thread(end_event, hw_queue) -> None:
  pm = messaging.PubMaster(['deviceState'])
  sm = messaging.SubMaster(["peripheralState", "gpsLocationExternal", "selfdriveState", "pandaStates"], poll="pandaStates")

  count = 0

  onroad_conditions: dict[str, bool] = {
    "ignition": False,
    "not_onroad_cycle": True,
    "device_temp_good": True,
  }
  startup_conditions: dict[str, bool] = {}
  startup_conditions_prev: dict[str, bool] = {}

  off_ts: float | None = None
  started_ts: float | None = None
  started_seen = False
  startup_blocked_ts: float | None = None
  thermal_status = ThermalStatus.yellow

  last_hw_state = HardwareState(
    network_type=NetworkType.none,
    network_info=None,
    network_metered=False,
    network_strength=NetworkStrength.unknown,
    network_stats={'wwanTx': -1, 'wwanRx': -1},
    modem_temps=[],
  )

  all_temp_filter = FirstOrderFilter(0., TEMP_TAU, DT_HW, initialized=False)
  offroad_temp_filter = FirstOrderFilter(0., TEMP_TAU, DT_HW, initialized=False)
  should_start_prev = False
  in_car = False
  engaged_prev = False
  offroad_cycle_count = 0

  params = Params()
  power_monitor = PowerMonitoring()

  uptime_offroad: float = params.get("UptimeOffroad", return_default=True)
  uptime_onroad: float = params.get("UptimeOnroad", return_default=True)
  last_uptime_ts: float = time.monotonic()

  HARDWARE.initialize_hardware()
  thermal_config = HARDWARE.get_thermal_config()

  fan_controller = None

  while not end_event.is_set():
    sm.update(PANDA_STATES_TIMEOUT)

    pandaStates = sm['pandaStates']
    peripheralState = sm['peripheralState']
    peripheral_panda_present = peripheralState.pandaType != log.PandaState.PandaType.unknown

    # handle requests to cycle system started state
    if params.get_bool("OnroadCycleRequested"):
      params.put_bool("OnroadCycleRequested", False)
      offroad_cycle_count = sm.frame
    onroad_conditions["not_onroad_cycle"] = (sm.frame - offroad_cycle_count) >= ONROAD_CYCLE_TIME * SERVICE_LIST['pandaStates'].frequency

    if sm.updated['pandaStates'] and len(pandaStates) > 0:

      # Set ignition based on any panda connected
      onroad_conditions["ignition"] = any(ps.ignitionLine or ps.ignitionCan for ps in pandaStates if ps.pandaType != log.PandaState.PandaType.unknown)

      pandaState = pandaStates[0]

      in_car = pandaState.harnessStatus != log.PandaState.HarnessStatus.notConnected

      # Setup fan handler on first connect to panda
      if fan_controller is None and peripheral_panda_present:
        if TICI:
          fan_controller = TiciFanController()

    elif (time.monotonic() - sm.recv_time['pandaStates']) > DISCONNECT_TIMEOUT:
      if onroad_conditions["ignition"]:
        onroad_conditions["ignition"] = False
        cloudlog.error("panda timed out onroad")

    # Run at 2Hz, plus either edge of ignition
    ign_edge = (started_ts is not None) != all(onroad_conditions.values())
    if (sm.frame % round(SERVICE_LIST['pandaStates'].frequency * DT_HW) != 0) and not ign_edge:
      continue

    msg = messaging.new_message('deviceState', valid=True)
    msg.deviceState = thermal_config.get_msg()
    msg.deviceState.deviceType = HARDWARE.get_device_type()

    try:
      last_hw_state = hw_queue.get_nowait()
    except queue.Empty:
      pass

    msg.deviceState.freeSpacePercent = get_available_percent(default=100.0)
    msg.deviceState.memoryUsagePercent = int(round(psutil.virtual_memory().percent))
    msg.deviceState.gpuUsagePercent = int(round(HARDWARE.get_gpu_usage_percent()))
    online_cpu_usage = [int(round(n)) for n in psutil.cpu_percent(percpu=True)]
    offline_cpu_usage = [0., ] * (len(msg.deviceState.cpuTempC) - len(online_cpu_usage))
    msg.deviceState.cpuUsagePercent = online_cpu_usage + offline_cpu_usage

    msg.deviceState.networkType = last_hw_state.network_type
    msg.deviceState.networkMetered = last_hw_state.network_metered
    msg.deviceState.networkStrength = last_hw_state.network_strength
    msg.deviceState.networkStats = last_hw_state.network_stats
    if last_hw_state.network_info is not None:
      msg.deviceState.networkInfo = last_hw_state.network_info

    msg.deviceState.modemTempC = last_hw_state.modem_temps

    msg.deviceState.screenBrightnessPercent = HARDWARE.get_screen_brightness()

    # this subset is only used for offroad
    temp_sources = [
      msg.deviceState.memoryTempC,
      max(msg.deviceState.cpuTempC, default=0.),
      max(msg.deviceState.gpuTempC, default=0.),
    ]
    offroad_comp_temp = offroad_temp_filter.update(max(temp_sources))

    # this drives the thermal status while onroad
    temp_sources.append(max(msg.deviceState.pmicTempC, default=0.))
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

    startup_conditions["up_to_date"] = params.get("Offroad_ConnectivityNeeded") is None or params.get_bool("DisableUpdates") or params.get_bool("SnoozeUpdate")
    startup_conditions["no_excessive_actuation"] = params.get("Offroad_ExcessiveActuation") is None
    startup_conditions["not_uninstalling"] = not params.get_bool("DoUninstall")
    startup_conditions["accepted_terms"] = params.get("HasAcceptedTerms") == terms_version

    # with 2% left, we killall, otherwise the phone will take a long time to boot
    startup_conditions["free_space"] = msg.deviceState.freeSpacePercent > 2
    startup_conditions["completed_training"] = params.get("CompletedTrainingVersion") == training_version
    startup_conditions["not_driver_view"] = not params.get_bool("IsDriverViewEnabled")
    startup_conditions["not_taking_snapshot"] = not params.get_bool("IsTakingSnapshot")

    # must be at an engageable thermal band to go onroad
    startup_conditions["device_temp_engageable"] = thermal_status < ThermalStatus.red

    # ensure device is fully booted
    startup_conditions["device_booted"] = startup_conditions.get("device_booted", False) or HARDWARE.booted()

    # if the temperature enters the danger zone, go offroad to cool down
    onroad_conditions["device_temp_good"] = thermal_status < ThermalStatus.danger
    extra_text = f"{offroad_comp_temp:.1f}C"
    show_alert = (not onroad_conditions["device_temp_good"] or not startup_conditions["device_temp_engageable"]) and onroad_conditions["ignition"]
    set_offroad_alert_if_changed("Offroad_TemperatureTooHigh", show_alert, extra_text=extra_text)

    # *** registration check ***
    if not PC:
      # we enforce this for our software, but you are welcome
      # to make a different decision in your software
      startup_conditions["registered_device"] = PC or (params.get("DongleId") != UNREGISTERED_DONGLE_ID)

    # TODO: this should move to TICI.initialize_hardware, but we currently can't import params there
    if TICI and HARDWARE.get_device_type() == "tici":
      if not os.path.isfile("/persist/comma/living-in-the-moment"):
        if not Path("/data/media").is_mount():
          set_offroad_alert_if_changed("Offroad_StorageMissing", True)

    # Handle offroad/onroad transition
    should_start = all(onroad_conditions.values())
    if started_ts is None:
      should_start = should_start and all(startup_conditions.values())

    if should_start != should_start_prev or (count == 0):
      params.put_bool("IsEngaged", False)
      engaged_prev = False
      HARDWARE.set_power_save(not should_start)

    if sm.updated['selfdriveState']:
      engaged = sm['selfdriveState'].enabled
      if engaged != engaged_prev:
        params.put_bool("IsEngaged", engaged)
        engaged_prev = engaged

      try:
        with open('/dev/kmsg', 'w') as kmsg:
          kmsg.write(f"<3>[hardware] engaged: {engaged}\n")
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
      msg.deviceState.lastAthenaPingTime = last_ping

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
    for i, temp in enumerate(msg.deviceState.pmicTempC):
      statlog.gauge(f"pmic{i}_temperature", temp)
    for i, temp in enumerate(last_hw_state.modem_temps):
      statlog.gauge(f"modem_temperature{i}", temp)
    statlog.gauge("fan_speed_percent_desired", msg.deviceState.fanSpeedPercentDesired)
    statlog.gauge("screen_brightness_percent", msg.deviceState.screenBrightnessPercent)

    # report to server once every 10 minutes
    rising_edge_started = should_start and not should_start_prev
    if rising_edge_started or (count % int(600. / DT_HW)) == 0:
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
          params.put("LastOffroadStatusPacket", dat)
        except Exception:
          cloudlog.exception("failed to save offroad status")

    params.put_bool_nonblocking("NetworkMetered", msg.deviceState.networkMetered)

    now_ts = time.monotonic()
    if off_ts:
      uptime_offroad += now_ts - max(last_uptime_ts, off_ts)
    elif started_ts:
      uptime_onroad += now_ts - max(last_uptime_ts, started_ts)
    last_uptime_ts = now_ts

    if (count % int(60. / DT_HW)) == 0:
      params.put("UptimeOffroad", uptime_offroad)
      params.put("UptimeOnroad", uptime_onroad)

    count += 1
    should_start_prev = should_start


def main():
  hw_queue = queue.Queue(maxsize=1)
  end_event = threading.Event()

  threads = [
    threading.Thread(target=hw_state_thread, args=(end_event, hw_queue)),
    threading.Thread(target=hardware_thread, args=(end_event, hw_queue)),
  ]

  if TICI:
    threads.append(threading.Thread(target=touch_thread, args=(end_event,)))

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
