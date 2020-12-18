import subprocess

from cereal import log
from selfdrive.hardware.base import HardwareBase

NM = 'org.freedesktop.NetworkManager'
NM_CON_ACT = NM + '.Connection.Active'
NM_DEV_WL = NM + '.Device.Wireless'
NM_AP = NM + '.AccessPoint'
DBUS_PROPS = 'org.freedesktop.DBus.Properties'

MM = 'org.freedesktop.ModemManager1'
MM_MODEM = MM + ".Modem"
MM_MODEM_SIMPLE = MM + ".Modem.Simple"
MM_SIM = MM + ".Sim"

MM_MODEM_STATE_CONNECTED = 11

NetworkType = log.ThermalData.NetworkType
NetworkStrength = log.ThermalData.NetworkStrength

# https://developer.gnome.org/ModemManager/unstable/ModemManager-Flags-and-Enumerations.html#MMModemAccessTechnology
MM_MODEM_ACCESS_TECHNOLOGY_UMTS = 1 << 5
MM_MODEM_ACCESS_TECHNOLOGY_LTE = 1 << 14


class Tici(HardwareBase):
  def __init__(self):
    import dbus  # pylint: disable=import-error

    self.bus = dbus.SystemBus()
    self.nm = self.bus.get_object(NM, '/org/freedesktop/NetworkManager')
    self.mm = self.bus.get_object(MM, '/org/freedesktop/ModemManager1')

  def get_sound_card_online(self):
    return True

  def reboot(self, reason=None):
    subprocess.check_output(["sudo", "reboot"])

  def uninstall(self):
    # TODO: implement uninstall. reboot to factory reset?
    pass

  def get_serial(self):
    return self.get_cmdline()['androidboot.serialno']

  def get_network_type(self):
    primary_connection = self.nm.Get(NM, 'PrimaryConnection', dbus_interface=DBUS_PROPS)
    primary_connection = self.bus.get_object(NM, primary_connection)
    tp = primary_connection.Get(NM_CON_ACT, 'Type', dbus_interface=DBUS_PROPS)

    if tp in ['802-3-ethernet', '802-11-wireless']:
      return NetworkType.wifi
    elif tp in ['gsm']:
      modem = self.get_modem()
      access_t = modem.Get(MM_MODEM, 'AccessTechnologies', dbus_interface=DBUS_PROPS)
      if access_t >= MM_MODEM_ACCESS_TECHNOLOGY_LTE:
        return NetworkType.cell4G
      elif access_t >= MM_MODEM_ACCESS_TECHNOLOGY_UMTS:
        return NetworkType.cell3G
      else:
        return NetworkType.cell2G

    return NetworkType.none

  def get_modem(self):
    objects = self.mm.GetManagedObjects(dbus_interface="org.freedesktop.DBus.ObjectManager")
    modem_path = list(objects.keys())[0]
    return self.bus.get_object(MM, modem_path)

  def get_wlan(self):
    wlan_path = self.nm.GetDeviceByIpIface('wlan0', dbus_interface=NM)
    return self.bus.get_object(NM, wlan_path)

  def get_sim_info(self):
    modem = self.get_modem()
    sim_path = modem.Get(MM_MODEM, 'Sim', dbus_interface=DBUS_PROPS)

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
        'sim_id': str(sim.Get(MM_SIM, 'SimIdentifier', dbus_interface=DBUS_PROPS)),
        'mcc_mnc': str(sim.Get(MM_SIM, 'OperatorIdentifier', dbus_interface=DBUS_PROPS)),
        'network_type': ["Unknown"],
        'sim_state': ["READY"],
        'data_connected': modem.Get(MM_MODEM, 'State', dbus_interface=DBUS_PROPS) == MM_MODEM_STATE_CONNECTED,
      }

  def get_subscriber_info(self):
    return ""

  def get_imei(self, slot):
    if slot != 0:
      return ""

    return str(self.get_modem().Get(MM_MODEM, 'EquipmentIdentifier', dbus_interface=DBUS_PROPS))

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

    if network_type == NetworkType.none:
      pass
    elif network_type == NetworkType.wifi:
      wlan = self.get_wlan()
      active_ap_path = wlan.Get(NM_DEV_WL, 'ActiveAccessPoint', dbus_interface=DBUS_PROPS)
      if active_ap_path != "/":
        active_ap = self.bus.get_object(NM, active_ap_path)
        strength = int(active_ap.Get(NM_AP, 'Strength', dbus_interface=DBUS_PROPS))
        network_strength = self.parse_strength(strength)
    else: # Cellular
      modem = self.get_modem()
      strength = int(modem.Get(MM_MODEM, 'SignalQuality', dbus_interface=DBUS_PROPS)[0])
      network_strength = self.parse_strength(strength)

    return network_strength

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
