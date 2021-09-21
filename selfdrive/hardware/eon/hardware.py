import binascii
import itertools
import os
import re
import serial
import struct
import subprocess
from typing import List, Union

from cereal import log
from selfdrive.hardware.base import HardwareBase, ThermalConfig

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

MODEM_PATH = "/dev/smd11"

def service_call(call: List[str]) -> Union[bytes, None]:
  try:
    ret = subprocess.check_output(["service", "call", *call], encoding='utf8').strip()
    if 'Parcel' not in ret:
      return None
    return parse_service_call_bytes(ret)
  except subprocess.CalledProcessError:
    return None


def parse_service_call_unpack(r, fmt) -> Union[bytes, None]:
  try:
    return struct.unpack(fmt, r)[0]
  except Exception:
    return None


def parse_service_call_string(r: bytes) -> Union[str, None]:
  try:
    r = r[8:]  # Cut off length field
    r_str = r.decode('utf_16_be')

    # All pairs of two characters seem to be swapped. Not sure why
    result = ""
    for a, b, in itertools.zip_longest(r_str[::2], r_str[1::2], fillvalue='\x00'):
      result += b + a

    return result.replace('\x00', '')
  except Exception:
    return None


def parse_service_call_bytes(ret: str) -> Union[bytes, None]:
  try:
    r = b""
    for hex_part in re.findall(r'[ (]([0-9a-f]{8})', ret):
      r += binascii.unhexlify(hex_part)
    return r
  except Exception:
    return None


def getprop(key: str) -> Union[str, None]:
  try:
    return subprocess.check_output(["getprop", key], encoding='utf8').strip()
  except subprocess.CalledProcessError:
    return None


class Android(HardwareBase):
  def get_os_version(self):
    with open("/VERSION") as f:
      return f.read().strip()

  def get_device_type(self):
    return "eon"

  def get_sound_card_online(self):
    return (os.path.isfile('/proc/asound/card0/state') and
            open('/proc/asound/card0/state').read().strip() == 'ONLINE')

  def get_imei(self, slot):
    slot = str(slot)
    if slot not in ("0", "1"):
      raise ValueError("SIM slot must be 0 or 1")

    return parse_service_call_string(service_call(["iphonesubinfo", "3", "i32", str(slot)]))

  def get_serial(self):
    ret = getprop("ro.serialno")
    if len(ret) == 0:
      ret = "cccccccc"
    return ret

  def get_subscriber_info(self):
    ret = parse_service_call_string(service_call(["iphonesubinfo", "7"]))
    if ret is None or len(ret) < 8:
      return ""
    return ret

  def reboot(self, reason=None):
    # e.g. reason="recovery" to go into recover mode
    if reason is None:
      reason_args = ["null"]
    else:
      reason_args = ["s16", reason]

    subprocess.check_output([
      "service", "call", "power", "16",  # IPowerManager.reboot
      "i32", "0",  # no confirmation,
      *reason_args,
      "i32", "1"  # wait
    ])

  def uninstall(self):
    with open('/cache/recovery/command', 'w') as f:
      f.write('--wipe_data\n')
    # IPowerManager.reboot(confirm=false, reason="recovery", wait=true)
    self.reboot(reason="recovery")

  def get_sim_info(self):
    # Used for athena
    # TODO: build using methods from this class
    sim_state = getprop("gsm.sim.state").split(",")
    network_type = getprop("gsm.network.type").split(',')
    mcc_mnc = getprop("gsm.sim.operator.numeric") or None

    sim_id = parse_service_call_string(service_call(['iphonesubinfo', '11']))
    cell_data_state = parse_service_call_unpack(service_call(['phone', '46']), ">q")
    cell_data_connected = (cell_data_state == 2)

    return {
      'sim_id': sim_id,
      'mcc_mnc': mcc_mnc,
      'network_type': network_type,
      'sim_state': sim_state,
      'data_connected': cell_data_connected
    }

  def get_network_info(self):
    msg = log.DeviceState.NetworkInfo.new_message()
    msg.state = getprop("gsm.sim.state") or ""
    msg.technology = getprop("gsm.network.type") or ""
    msg.operator = getprop("gsm.sim.operator.numeric") or ""

    try:
      modem = serial.Serial(MODEM_PATH, 115200, timeout=0.1)
      modem.write(b"AT$QCRSRP?\r")
      msg.extra = modem.read_until(b"OK\r\n").decode('utf-8')

      rsrp = msg.extra.split("$QCRSRP: ")[1].split("\r")[0].split(",")
      msg.channel = int(rsrp[1])
    except Exception:
      pass

    return msg

  def get_network_type(self):
    wifi_check = parse_service_call_string(service_call(["connectivity", "2"]))
    if wifi_check is None:
      return NetworkType.none
    elif 'WIFI' in wifi_check:
      return NetworkType.wifi
    else:
      cell_check = parse_service_call_unpack(service_call(['phone', '59']), ">q")
      # from TelephonyManager.java
      cell_networks = {
        0: NetworkType.none,
        1: NetworkType.cell2G,
        2: NetworkType.cell2G,
        3: NetworkType.cell3G,
        4: NetworkType.cell2G,
        5: NetworkType.cell3G,
        6: NetworkType.cell3G,
        7: NetworkType.cell3G,
        8: NetworkType.cell3G,
        9: NetworkType.cell3G,
        10: NetworkType.cell3G,
        11: NetworkType.cell2G,
        12: NetworkType.cell3G,
        13: NetworkType.cell4G,
        14: NetworkType.cell4G,
        15: NetworkType.cell3G,
        16: NetworkType.cell2G,
        17: NetworkType.cell3G,
        18: NetworkType.cell4G,
        19: NetworkType.cell4G
      }
      return cell_networks.get(cell_check, NetworkType.none)

  def get_network_strength(self, network_type):
    network_strength = NetworkStrength.unknown

    # from SignalStrength.java
    def get_lte_level(rsrp, rssnr):
      INT_MAX = 2147483647
      if rsrp == INT_MAX:
        lvl_rsrp = NetworkStrength.unknown
      elif rsrp >= -95:
        lvl_rsrp = NetworkStrength.great
      elif rsrp >= -105:
        lvl_rsrp = NetworkStrength.good
      elif rsrp >= -115:
        lvl_rsrp = NetworkStrength.moderate
      else:
        lvl_rsrp = NetworkStrength.poor
      if rssnr == INT_MAX:
        lvl_rssnr = NetworkStrength.unknown
      elif rssnr >= 45:
        lvl_rssnr = NetworkStrength.great
      elif rssnr >= 10:
        lvl_rssnr = NetworkStrength.good
      elif rssnr >= -30:
        lvl_rssnr = NetworkStrength.moderate
      else:
        lvl_rssnr = NetworkStrength.poor
      return max(lvl_rsrp, lvl_rssnr)

    def get_tdscdma_level(tdscmadbm):
      lvl = NetworkStrength.unknown
      if tdscmadbm > -25:
        lvl = NetworkStrength.unknown
      elif tdscmadbm >= -49:
        lvl = NetworkStrength.great
      elif tdscmadbm >= -73:
        lvl = NetworkStrength.good
      elif tdscmadbm >= -97:
        lvl = NetworkStrength.moderate
      elif tdscmadbm >= -110:
        lvl = NetworkStrength.poor
      return lvl

    def get_gsm_level(asu):
      if asu <= 2 or asu == 99:
        lvl = NetworkStrength.unknown
      elif asu >= 12:
        lvl = NetworkStrength.great
      elif asu >= 8:
        lvl = NetworkStrength.good
      elif asu >= 5:
        lvl = NetworkStrength.moderate
      else:
        lvl = NetworkStrength.poor
      return lvl

    def get_evdo_level(evdodbm, evdosnr):
      lvl_evdodbm = NetworkStrength.unknown
      lvl_evdosnr = NetworkStrength.unknown
      if evdodbm >= -65:
        lvl_evdodbm = NetworkStrength.great
      elif evdodbm >= -75:
        lvl_evdodbm = NetworkStrength.good
      elif evdodbm >= -90:
        lvl_evdodbm = NetworkStrength.moderate
      elif evdodbm >= -105:
        lvl_evdodbm = NetworkStrength.poor
      if evdosnr >= 7:
        lvl_evdosnr = NetworkStrength.great
      elif evdosnr >= 5:
        lvl_evdosnr = NetworkStrength.good
      elif evdosnr >= 3:
        lvl_evdosnr = NetworkStrength.moderate
      elif evdosnr >= 1:
        lvl_evdosnr = NetworkStrength.poor
      return max(lvl_evdodbm, lvl_evdosnr)

    def get_cdma_level(cdmadbm, cdmaecio):
      lvl_cdmadbm = NetworkStrength.unknown
      lvl_cdmaecio = NetworkStrength.unknown
      if cdmadbm >= -75:
        lvl_cdmadbm = NetworkStrength.great
      elif cdmadbm >= -85:
        lvl_cdmadbm = NetworkStrength.good
      elif cdmadbm >= -95:
        lvl_cdmadbm = NetworkStrength.moderate
      elif cdmadbm >= -100:
        lvl_cdmadbm = NetworkStrength.poor
      if cdmaecio >= -90:
        lvl_cdmaecio = NetworkStrength.great
      elif cdmaecio >= -110:
        lvl_cdmaecio = NetworkStrength.good
      elif cdmaecio >= -130:
        lvl_cdmaecio = NetworkStrength.moderate
      elif cdmaecio >= -150:
        lvl_cdmaecio = NetworkStrength.poor
      return max(lvl_cdmadbm, lvl_cdmaecio)

    if network_type == NetworkType.none:
      return network_strength
    if network_type == NetworkType.wifi:
      out = subprocess.check_output('dumpsys connectivity', shell=True).decode('utf-8')
      network_strength = NetworkStrength.unknown
      for line in out.split('\n'):
        signal_str = "SignalStrength: "
        if signal_str in line:
          lvl_idx_start = line.find(signal_str) + len(signal_str)
          lvl_idx_end = line.find(']', lvl_idx_start)
          lvl = int(line[lvl_idx_start : lvl_idx_end])
          if lvl >= -50:
            network_strength = NetworkStrength.great
          elif lvl >= -60:
            network_strength = NetworkStrength.good
          elif lvl >= -70:
            network_strength = NetworkStrength.moderate
          else:
            network_strength = NetworkStrength.poor
      return network_strength
    else:
      # check cell strength
      out = subprocess.check_output('dumpsys telephony.registry', shell=True).decode('utf-8')
      for line in out.split('\n'):
        if "mSignalStrength" in line:
          arr = line.split(' ')
          ns = 0
          if ("gsm" in arr[14]):
            rsrp = int(arr[9])
            rssnr = int(arr[11])
            ns = get_lte_level(rsrp, rssnr)
            if ns == NetworkStrength.unknown:
              tdscmadbm = int(arr[13])
              ns = get_tdscdma_level(tdscmadbm)
              if ns == NetworkStrength.unknown:
                asu = int(arr[1])
                ns = get_gsm_level(asu)
          else:
            cdmadbm = int(arr[3])
            cdmaecio = int(arr[4])
            evdodbm = int(arr[5])
            evdosnr = int(arr[7])
            lvl_cdma = get_cdma_level(cdmadbm, cdmaecio)
            lvl_edmo = get_evdo_level(evdodbm, evdosnr)
            if lvl_edmo == NetworkStrength.unknown:
              ns = lvl_cdma
            elif lvl_cdma == NetworkStrength.unknown:
              ns = lvl_edmo
            else:
              ns = min(lvl_cdma, lvl_edmo)
          network_strength = max(network_strength, ns)

      return network_strength

  def get_battery_capacity(self):
    return self.read_param_file("/sys/class/power_supply/battery/capacity", int, 100)

  def get_battery_status(self):
    # This does not correspond with actual charging or not.
    # If a USB cable is plugged in, it responds with 'Charging', even when charging is disabled
    return self.read_param_file("/sys/class/power_supply/battery/status", lambda x: x.strip(), '')

  def get_battery_current(self):
    return self.read_param_file("/sys/class/power_supply/battery/current_now", int)

  def get_battery_voltage(self):
    return self.read_param_file("/sys/class/power_supply/battery/voltage_now", int)

  def get_battery_charging(self):
    # This does correspond with actually charging
    return self.read_param_file("/sys/class/power_supply/battery/charge_type", lambda x: x.strip() != "N/A", True)

  def set_battery_charging(self, on):
    with open('/sys/class/power_supply/battery/charging_enabled', 'w') as f:
      f.write(f"{1 if on else 0}\n")

  def get_usb_present(self):
    return self.read_param_file("/sys/class/power_supply/usb/present", lambda x: bool(int(x)), False)

  def get_current_power_draw(self):
    # We don't have a good direct way to measure this on android
    return None

  def shutdown(self):
    os.system('LD_LIBRARY_PATH="" svc power shutdown')

  def get_thermal_config(self):
    return ThermalConfig(cpu=((5, 7, 10, 12), 10), gpu=((16,), 10), mem=(2, 10), bat=(29, 1000), ambient=(25, 1))

  def set_screen_brightness(self, percentage):
    with open("/sys/class/leds/lcd-backlight/brightness", "w") as f:
      f.write(str(int(percentage * 2.55)))

  def set_power_save(self, powersave_enabled):
    pass

  def get_gpu_usage_percent(self):
    try:
      used, total = open('/sys/devices/soc/b00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/gpubusy').read().strip().split()
      perc = 100.0 * int(used) / int(total)
      return min(max(perc, 0), 100)
    except Exception:
      return 0

  def get_modem_version(self):
    return None

  def get_modem_temperatures(self):
    # Not sure if we can get this on the LeEco
    return []

  def initialize_hardware(self):
    pass

  def get_networks(self):
    return None
