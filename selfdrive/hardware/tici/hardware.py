import json
import os
import subprocess
from functools import cached_property
from enum import IntEnum
from pathlib import Path

from cereal import log
from selfdrive.hardware.base import HardwareBase, ThermalConfig
from selfdrive.hardware.tici.amplifier import Amplifier
from selfdrive.hardware.tici import iwlist

NM = 'org.freedesktop.NetworkManager'
NM_CON_ACT = NM + '.Connection.Active'
NM_DEV_WL = NM + '.Device.Wireless'
NM_AP = NM + '.AccessPoint'
DBUS_PROPS = 'org.freedesktop.DBus.Properties'

MM = 'org.freedesktop.ModemManager1'
MM_MODEM = MM + ".Modem"
MM_MODEM_SIMPLE = MM + ".Modem.Simple"
MM_SIM = MM + ".Sim"

class MM_MODEM_STATE(IntEnum):
       FAILED        = -1
       UNKNOWN       = 0
       INITIALIZING  = 1
       LOCKED        = 2
       DISABLED      = 3
       DISABLING     = 4
       ENABLING      = 5
       ENABLED       = 6
       SEARCHING     = 7
       REGISTERED    = 8
       DISCONNECTING = 9
       CONNECTING    = 10
       CONNECTED     = 11

TIMEOUT = 0.1

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

# https://developer.gnome.org/ModemManager/unstable/ModemManager-Flags-and-Enumerations.html#MMModemAccessTechnology
MM_MODEM_ACCESS_TECHNOLOGY_UMTS = 1 << 5
MM_MODEM_ACCESS_TECHNOLOGY_LTE = 1 << 14

class Tici(HardwareBase):
  @cached_property
  def bus(self):
    import dbus  # pylint: disable=import-error
    return dbus.SystemBus()

  @cached_property
  def nm(self):
    return self.bus.get_object(NM, '/org/freedesktop/NetworkManager')

  @cached_property
  def mm(self):
    return self.bus.get_object(MM, '/org/freedesktop/ModemManager1')

  @cached_property
  def amplifier(self):
    return Amplifier()

  def get_os_version(self):
    with open("/VERSION") as f:
      return f.read().strip()

  def get_device_type(self):
    return "tici"

  def get_sound_card_online(self):
    return (os.path.isfile('/proc/asound/card0/state') and
            open('/proc/asound/card0/state').read().strip() == 'ONLINE')

  def reboot(self, reason=None):
    subprocess.check_output(["sudo", "reboot"])

  def uninstall(self):
    Path("/data/__system_reset__").touch()
    os.sync()
    self.reboot()

  def get_serial(self):
    return self.get_cmdline()['androidboot.serialno']

  def get_network_type(self):
    try:
      primary_connection = self.nm.Get(NM, 'PrimaryConnection', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
      primary_connection = self.bus.get_object(NM, primary_connection)
      primary_type = primary_connection.Get(NM_CON_ACT, 'Type', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
      primary_id = primary_connection.Get(NM_CON_ACT, 'Id', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)

      if primary_type == '802-3-ethernet':
        return NetworkType.ethernet
      elif primary_type == '802-11-wireless' and primary_id != 'Hotspot':
        return NetworkType.wifi
      else:
        active_connections = self.nm.Get(NM, 'ActiveConnections', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
        for conn in active_connections:
          c = self.bus.get_object(NM, conn)
          tp = c.Get(NM_CON_ACT, 'Type', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
          if tp == 'gsm':
            modem = self.get_modem()
            access_t = modem.Get(MM_MODEM, 'AccessTechnologies', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
            if access_t >= MM_MODEM_ACCESS_TECHNOLOGY_LTE:
              return NetworkType.cell4G
            elif access_t >= MM_MODEM_ACCESS_TECHNOLOGY_UMTS:
              return NetworkType.cell3G
            else:
              return NetworkType.cell2G
    except Exception:
      pass

    return NetworkType.none

  def get_modem(self):
    objects = self.mm.GetManagedObjects(dbus_interface="org.freedesktop.DBus.ObjectManager", timeout=TIMEOUT)
    modem_path = list(objects.keys())[0]
    return self.bus.get_object(MM, modem_path)

  def get_wlan(self):
    wlan_path = self.nm.GetDeviceByIpIface('wlan0', dbus_interface=NM, timeout=TIMEOUT)
    return self.bus.get_object(NM, wlan_path)

  def get_sim_info(self):
    modem = self.get_modem()
    sim_path = modem.Get(MM_MODEM, 'Sim', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)

    if sim_path == "/":
      return {
        'sim_id': '',
        'mcc_mnc': None,
        'network_type': ["Unknown"],
        'sim_state': ["ABSENT"],
        'data_connected': False
      }
    else:
      sim = self.bus.get_object(MM, sim_path)
      return {
        'sim_id': str(sim.Get(MM_SIM, 'SimIdentifier', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)),
        'mcc_mnc': str(sim.Get(MM_SIM, 'OperatorIdentifier', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)),
        'network_type': ["Unknown"],
        'sim_state': ["READY"],
        'data_connected': modem.Get(MM_MODEM, 'State', dbus_interface=DBUS_PROPS, timeout=TIMEOUT) == MM_MODEM_STATE.CONNECTED,
      }

  def get_subscriber_info(self):
    return ""

  def get_imei(self, slot):
    if slot != 0:
      return ""

    return str(self.get_modem().Get(MM_MODEM, 'EquipmentIdentifier', dbus_interface=DBUS_PROPS, timeout=TIMEOUT))

  def get_network_info(self):
    modem = self.get_modem()
    try:
      info = modem.Command("AT+QNWINFO", int(TIMEOUT * 1000), dbus_interface=MM_MODEM, timeout=TIMEOUT)
      extra = modem.Command('AT+QENG="servingcell"', int(TIMEOUT * 1000), dbus_interface=MM_MODEM, timeout=TIMEOUT)
      state = modem.Get(MM_MODEM, 'State', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
    except Exception:
      return None

    if info and info.startswith('+QNWINFO: '):
      info = info.replace('+QNWINFO: ', '').replace('"', '').split(',')
      extra = "" if extra is None else extra.replace('+QENG: "servingcell",', '').replace('"', '')
      state = "" if state is None else MM_MODEM_STATE(state).name

      if len(info) != 4:
        return None

      technology, operator, band, channel = info

      return({
        'technology': technology,
        'operator': operator,
        'band': band,
        'channel': int(channel),
        'extra': extra,
        'state': state,
      })
    else:
      return None

  def parse_strength(self, percentage):
      if percentage < 25:
        return NetworkStrength.poor
      elif percentage < 50:
        return NetworkStrength.moderate
      elif percentage < 75:
        return NetworkStrength.good
      else:
        return NetworkStrength.great

  def get_network_strength(self, network_type):
    network_strength = NetworkStrength.unknown

    try:
      if network_type == NetworkType.none:
        pass
      elif network_type == NetworkType.wifi:
        wlan = self.get_wlan()
        active_ap_path = wlan.Get(NM_DEV_WL, 'ActiveAccessPoint', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
        if active_ap_path != "/":
          active_ap = self.bus.get_object(NM, active_ap_path)
          strength = int(active_ap.Get(NM_AP, 'Strength', dbus_interface=DBUS_PROPS, timeout=TIMEOUT))
          network_strength = self.parse_strength(strength)
      else:  # Cellular
        modem = self.get_modem()
        strength = int(modem.Get(MM_MODEM, 'SignalQuality', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)[0])
        network_strength = self.parse_strength(strength)
    except Exception:
      pass

    return network_strength

  def get_modem_version(self):
    try:
      modem = self.get_modem()
      return modem.Get(MM_MODEM, 'Revision', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
    except Exception:
      return None

  def get_modem_temperatures(self):
    modem = self.get_modem()
    try:
      command_timeout = 0.2
      temps = modem.Command("AT+QTEMP", int(command_timeout * 1000), dbus_interface=MM_MODEM, timeout=command_timeout)
      return list(map(int, temps.split(' ')[1].split(',')))
    except Exception:
      return []

  def get_nvme_temperatures(self):
    ret = []
    try:
      out = subprocess.check_output("sudo smartctl -aj /dev/nvme0", shell=True)
      dat = json.loads(out)
      ret = list(map(int, dat["nvme_smart_health_information_log"]["temperature_sensors"]))
    except Exception:
      pass
    return ret

  # We don't have a battery, so let's use some sane constants
  def get_battery_capacity(self):
    return 100

  def get_battery_status(self):
    return ""

  def get_battery_current(self):
    return 0

  def get_battery_voltage(self):
    return 0

  def get_battery_charging(self):
    return True

  def set_battery_charging(self, on):
    pass

  def get_usb_present(self):
    # Not sure if relevant on tici, but the file exists
    return self.read_param_file("/sys/class/power_supply/usb/present", lambda x: bool(int(x)), False)

  def get_current_power_draw(self):
    return (self.read_param_file("/sys/class/hwmon/hwmon1/power1_input", int) / 1e6)

  def shutdown(self):
    # Note that for this to work and have the device stay powered off, the panda needs to be in UsbPowerMode::CLIENT!
    os.system("sudo poweroff")

  def get_thermal_config(self):
    return ThermalConfig(cpu=((1, 2, 3, 4, 5, 6, 7, 8), 1000), gpu=((48,49), 1000), mem=(15, 1000), bat=(None, 1), ambient=(65, 1000))

  def set_screen_brightness(self, percentage):
    try:
      with open("/sys/class/backlight/panel0-backlight/brightness", "w") as f:
        f.write(str(int(percentage * 10.23)))
    except Exception:
      pass

  def set_power_save(self, powersave_enabled):
    # amplifier, 100mW at idle
    self.amplifier.set_global_shutdown(amp_disabled=powersave_enabled)
    if not powersave_enabled:
      self.amplifier.initialize_configuration()

    # offline big cluster, leave core 4 online for boardd
    for i in range(5, 8):
      # TODO: fix permissions with udev
      val = "0" if powersave_enabled else "1"
      os.system(f"sudo su -c 'echo {val} > /sys/devices/system/cpu/cpu{i}/online'")

  def get_gpu_usage_percent(self):
    try:
      used, total = open('/sys/class/kgsl/kgsl-3d0/gpubusy').read().strip().split()
      return 100.0 * int(used) / int(total)
    except Exception:
      return 0

  def initialize_hardware(self):
    self.amplifier.initialize_configuration()

  def get_networks(self):
    r = {}

    wlan = iwlist.scan()
    if wlan is not None:
      r['wlan'] = wlan

    lte_info = self.get_network_info()
    if lte_info is not None:
      extra = lte_info['extra']

      # <state>,"LTE",<is_tdd>,<mcc>,<mnc>,<cellid>,<pcid>,<earfcn>,<freq_band_ind>,
      # <ul_bandwidth>,<dl_bandwidth>,<tac>,<rsrp>,<rsrq>,<rssi>,<sinr>,<srxlev>
      if 'LTE' in extra:
        extra = extra.split(',')
        try:
          r['lte'] = [{
            "mcc": int(extra[3]),
            "mnc": int(extra[4]),
            "cid": int(extra[5], 16),
            "nmr": [{"pci": int(extra[6]), "earfcn": int(extra[7])}],
          }]
        except (ValueError, IndexError):
          pass

    return r
