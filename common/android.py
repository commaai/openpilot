import os
import binascii
import itertools
import re
import struct
import subprocess
import random
from cereal import log

NetworkType = log.ThermalData.NetworkType
NetworkStrength = log.ThermalData.NetworkStrength

ANDROID = os.path.isfile('/EON')

def get_sound_card_online():
  return (os.path.isfile('/proc/asound/card0/state') and
          open('/proc/asound/card0/state').read().strip() == 'ONLINE')

def getprop(key):
  if not ANDROID:
    return ""
  return subprocess.check_output(["getprop", key], encoding='utf8').strip()

def get_imei(slot):
  slot = str(slot)
  if slot not in ("0", "1"):
    raise ValueError("SIM slot must be 0 or 1")

  ret = parse_service_call_string(service_call(["iphonesubinfo", "3" , "i32", str(slot)]))
  if not ret:
    # allow non android to be identified differently
    ret = "%015d" % random.randint(0, 1 << 32)
  return ret

def get_serial():
  ret = getprop("ro.serialno")
  if ret == "":
    ret = "cccccccc"
  return ret

def get_subscriber_info():
  ret = parse_service_call_string(service_call(["iphonesubinfo", "7"]))
  if ret is None or len(ret) < 8:
    return ""
  return ret

def reboot(reason=None):
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

def service_call(call):
  if not ANDROID:
    return None

  ret = subprocess.check_output(["service", "call", *call], encoding='utf8').strip()
  if 'Parcel' not in ret:
    return None

  return parse_service_call_bytes(ret)

def parse_service_call_unpack(r, fmt):
  try:
    return struct.unpack(fmt, r)[0]
  except Exception:
    return None

def parse_service_call_string(r):
  try:
    r = r[8:]  # Cut off length field
    r = r.decode('utf_16_be')

    # All pairs of two characters seem to be swapped. Not sure why
    result = ""
    for a, b, in itertools.zip_longest(r[::2], r[1::2], fillvalue='\x00'):
        result += b + a

    result = result.replace('\x00', '')

    return result
  except Exception:
    return None

def parse_service_call_bytes(ret):
  try:
    r = b""
    for hex_part in re.findall(r'[ (]([0-9a-f]{8})', ret):
      r += binascii.unhexlify(hex_part)
    return r
  except Exception:
    return None

def get_network_type():
  if not ANDROID:
    return NetworkType.none

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

def get_network_strength(network_type):
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
