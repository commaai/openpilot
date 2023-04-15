import json
import math
import os
import subprocess
import time
from enum import IntEnum
from functools import cached_property
from pathlib import Path

from cereal import log
from common.gpio import gpio_set, gpio_init, get_irq_for_action
from system.hardware.base import HardwareBase, ThermalConfig
from system.hardware.tici import iwlist
from system.hardware.tici.pins import GPIO
from system.hardware.tici.amplifier import Amplifier

NM = 'org.freedesktop.NetworkManager'
NM_CON_ACT = NM + '.Connection.Active'
NM_DEV = NM + '.Device'
NM_DEV_WL = NM + '.Device.Wireless'
NM_DEV_STATS = NM + '.Device.Statistics'
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

class NMMetered(IntEnum):
  NM_METERED_UNKNOWN = 0
  NM_METERED_YES = 1
  NM_METERED_NO = 2
  NM_METERED_GUESS_YES = 3
  NM_METERED_GUESS_NO = 4

TIMEOUT = 0.1
REFRESH_RATE_MS = 1000

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

# https://developer.gnome.org/ModemManager/unstable/ModemManager-Flags-and-Enumerations.html#MMModemAccessTechnology
MM_MODEM_ACCESS_TECHNOLOGY_UMTS = 1 << 5
MM_MODEM_ACCESS_TECHNOLOGY_LTE = 1 << 14


def sudo_write(val, path):
  os.system(f"sudo su -c 'echo {val} > {path}'")


def affine_irq(val, action):
  irq = get_irq_for_action(action)
  if len(irq) == 0:
    print(f"No IRQs found for '{action}'")
    return
  for i in irq:
    sudo_write(str(val), f"/proc/irq/{i}/smp_affinity_list")


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

  @cached_property
  def model(self):
    with open("/sys/firmware/devicetree/base/model") as f:
      model = f.read().strip('\x00')
    model = model.split('comma ')[-1]
    # TODO: remove this with AGNOS 7+
    if model.startswith('Qualcomm'):
      model = 'tici'
    return model

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

      if primary_type == '802-3-ethernet':
        return NetworkType.ethernet
      elif primary_type == '802-11-wireless':
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

  def get_wwan(self):
    wwan_path = self.nm.GetDeviceByIpIface('wwan0', dbus_interface=NM, timeout=TIMEOUT)
    return self.bus.get_object(NM, wwan_path)

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
      info = modem.Command("AT+QNWINFO", math.ceil(TIMEOUT), dbus_interface=MM_MODEM, timeout=TIMEOUT)
      extra = modem.Command('AT+QENG="servingcell"', math.ceil(TIMEOUT), dbus_interface=MM_MODEM, timeout=TIMEOUT)
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

  def get_network_metered(self, network_type) -> bool:
    try:
      primary_connection = self.nm.Get(NM, 'PrimaryConnection', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
      primary_connection = self.bus.get_object(NM, primary_connection)
      primary_devices = primary_connection.Get(NM_CON_ACT, 'Devices', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)

      for dev in primary_devices:
        dev_obj = self.bus.get_object(NM, str(dev))
        metered_prop = dev_obj.Get(NM_DEV, 'Metered', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)

        if network_type == NetworkType.wifi:
          if metered_prop in [NMMetered.NM_METERED_YES, NMMetered.NM_METERED_GUESS_YES]:
            return True
        elif network_type in [NetworkType.cell2G, NetworkType.cell3G, NetworkType.cell4G, NetworkType.cell5G]:
          if metered_prop == NMMetered.NM_METERED_NO:
            return False
    except Exception:
      pass

    return super().get_network_metered(network_type)

  @staticmethod
  def set_bandwidth_limit(upload_speed_kbps: int, download_speed_kbps: int) -> None:
    upload_speed_kbps = int(upload_speed_kbps)  # Ensure integer value
    download_speed_kbps = int(download_speed_kbps)  # Ensure integer value

    adapter = "wwan0"
    ifb = "ifb0"

    sudo = ["sudo"]
    tc = sudo + ["tc"]

    # check, cmd
    cleanup = [
      # Clean up old rules
      (False, tc + ["qdisc", "del", "dev", adapter, "root"]),
      (False, tc + ["qdisc", "del", "dev", ifb, "root"]),
      (False, tc + ["qdisc", "del", "dev", adapter, "ingress"]),
      (False, tc + ["qdisc", "del", "dev", ifb, "ingress"]),

      # Bring ifb0 down
      (False, sudo + ["ip", "link", "set", "dev", ifb, "down"]),
    ]

    upload = [
      # Create root Hierarchy Token Bucket that sends all traffic to 1:20
      (True, tc + ["qdisc", "add", "dev", adapter, "root", "handle", "1:", "htb", "default", "20"]),

      # Create class 1:20 with specified rate limit
      (True, tc + ["class", "add", "dev", adapter, "parent", "1:", "classid", "1:20", "htb", "rate", f"{upload_speed_kbps}kbit"]),

      # Create universal 32 bit filter on adapter that sends all outbound ip traffic through the class
      (True, tc + ["filter", "add", "dev", adapter, "parent", "1:", "protocol", "ip", "prio", "10", "u32", "match", "ip", "dst", "0.0.0.0/0", "flowid", "1:20"]),
    ]

    download = [
      # Bring ifb0 up
      (True, sudo + ["ip", "link", "set", "dev", ifb, "up"]),

      # Redirect ingress (incoming) to egress ifb0
      (True, tc + ["qdisc", "add", "dev", adapter, "handle", "ffff:", "ingress"]),
      (True, tc + ["filter", "add", "dev", adapter, "parent", "ffff:", "protocol", "ip", "u32", "match", "u32", "0", "0", "action", "mirred", "egress", "redirect", "dev", ifb]),

      # Add class and rules for virtual interface
      (True, tc + ["qdisc", "add", "dev", ifb, "root", "handle", "2:", "htb"]),
      (True, tc + ["class", "add", "dev", ifb, "parent", "2:", "classid", "2:1", "htb", "rate", f"{download_speed_kbps}kbit"]),

      # Add filter to rule for IP address
      (True, tc + ["filter", "add", "dev", ifb, "protocol", "ip", "parent", "2:", "prio", "1", "u32", "match", "ip", "src", "0.0.0.0/0", "flowid", "2:1"]),
    ]

    commands = cleanup
    if upload_speed_kbps != -1:
      commands += upload
    if download_speed_kbps != -1:
      commands += download

    for check, cmd in commands:
      subprocess.run(cmd, check=check)

  def get_modem_version(self):
    try:
      modem = self.get_modem()
      return modem.Get(MM_MODEM, 'Revision', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
    except Exception:
      return None

  def get_modem_nv(self):
    timeout = 0.2  # Default timeout is too short
    files = (
      '/nv/item_files/modem/mmode/ue_usage_setting',
      '/nv/item_files/ims/IMS_enable',
      '/nv/item_files/modem/mmode/sms_only',
    )
    try:
      modem = self.get_modem()
      return { fn: str(modem.Command(f'AT+QNVFR="{fn}"', math.ceil(timeout), dbus_interface=MM_MODEM, timeout=timeout)) for fn in files}
    except Exception:
      return None

  def get_modem_temperatures(self):
    timeout = 0.2  # Default timeout is too short
    try:
      modem = self.get_modem()
      temps = modem.Command("AT+QTEMP", math.ceil(timeout), dbus_interface=MM_MODEM, timeout=timeout)
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

  def get_usb_present(self):
    # Not sure if relevant on tici, but the file exists
    return self.read_param_file("/sys/class/power_supply/usb/present", lambda x: bool(int(x)), False)

  def get_current_power_draw(self):
    return (self.read_param_file("/sys/class/hwmon/hwmon1/power1_input", int) / 1e6)

  def get_som_power_draw(self):
    return (self.read_param_file("/sys/class/power_supply/bms/voltage_now", int) * self.read_param_file("/sys/class/power_supply/bms/current_now", int) / 1e12)

  def shutdown(self):
    os.system("sudo poweroff")

  def get_thermal_config(self):
    return ThermalConfig(cpu=(["cpu%d-silver-usr" % i for i in range(4)] +
                              ["cpu%d-gold-usr" % i for i in range(4)], 1000),
                         gpu=(("gpu0-usr", "gpu1-usr"), 1000),
                         mem=("ddr-usr", 1000),
                         bat=(None, 1),
                         ambient=("xo-therm-adc", 1000),
                         pmic=(("pm8998_tz", "pm8005_tz"), 1000))

  def set_screen_brightness(self, percentage):
    try:
      with open("/sys/class/backlight/panel0-backlight/max_brightness") as f:
        max_brightness = float(f.read().strip())

      val = int(percentage * (max_brightness / 100.))
      with open("/sys/class/backlight/panel0-backlight/brightness", "w") as f:
        f.write(str(val))
    except Exception:
      pass

  def get_screen_brightness(self):
    try:
      with open("/sys/class/backlight/panel0-backlight/max_brightness") as f:
        max_brightness = float(f.read().strip())

      with open("/sys/class/backlight/panel0-backlight/brightness") as f:
        return int(float(f.read()) / (max_brightness / 100.))
    except Exception:
      return 0

  def set_power_save(self, powersave_enabled):
    # amplifier, 100mW at idle
    self.amplifier.set_global_shutdown(amp_disabled=powersave_enabled)
    if not powersave_enabled:
      self.amplifier.initialize_configuration(self.model)

    # *** CPU config ***

    # offline big cluster, leave core 4 online for boardd
    for i in range(5, 8):
      val = '0' if powersave_enabled else '1'
      sudo_write(val, f'/sys/devices/system/cpu/cpu{i}/online')

    for n in ('0', '4'):
      gov = 'ondemand' if powersave_enabled else 'performance'
      sudo_write(gov, f'/sys/devices/system/cpu/cpufreq/policy{n}/scaling_governor')

    # *** IRQ config ***

    # GPU
    affine_irq(5, "kgsl-3d0")

    # boardd core
    affine_irq(4, "spi_geni")         # SPI
    affine_irq(4, "xhci-hcd:usb3")    # aux panda USB (or potentially anything else on USB)
    if "tici" in self.model:
      affine_irq(4, "xhci-hcd:usb1")  # internal panda USB

    # camerad core
    camera_irqs = ("cci", "cpas_camnoc", "cpas-cdm", "csid", "ife", "csid", "csid-lite", "ife-lite")
    for n in camera_irqs:
      affine_irq(5, n)

  def get_gpu_usage_percent(self):
    try:
      used, total = open('/sys/class/kgsl/kgsl-3d0/gpubusy').read().strip().split()
      return 100.0 * int(used) / int(total)
    except Exception:
      return 0

  def initialize_hardware(self):
    self.amplifier.initialize_configuration(self.model)

    # Allow thermald to write engagement status to kmsg
    os.system("sudo chmod a+w /dev/kmsg")

    # TODO: remove the if once agnos 7 ships
    # Ensure fan gpio is enabled so fan runs until shutdown, also turned on at boot by the ABL
    if os.path.exists('/sys/class/gpio/gpio49/'):
      gpio_init(GPIO.SOM_ST_IO, True)
      gpio_set(GPIO.SOM_ST_IO, 1)

    # *** IRQ config ***

    # move these off the default core
    affine_irq(1, "msm_drm")
    affine_irq(1, "msm_vidc")
    affine_irq(1, "i2c_geni")

    # mask off big cluster from default affinity
    sudo_write("f", "/proc/irq/default_smp_affinity")

    # *** GPU config ***
    # https://github.com/commaai/agnos-kernel-sdm845/blob/master/arch/arm64/boot/dts/qcom/sdm845-gpu.dtsi#L216
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/min_pwrlevel")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/max_pwrlevel")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/force_bus_on")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/force_clk_on")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/force_rail_on")
    sudo_write("1000000", "/sys/class/kgsl/kgsl-3d0/idle_timer")
    sudo_write("performance", "/sys/class/kgsl/kgsl-3d0/devfreq/governor")
    sudo_write("596", "/sys/class/kgsl/kgsl-3d0/max_clock_mhz")

    # setup governors
    sudo_write("performance", "/sys/class/devfreq/soc:qcom,cpubw/governor")
    sudo_write("performance", "/sys/class/devfreq/soc:qcom,memlat-cpu0/governor")
    sudo_write("performance", "/sys/class/devfreq/soc:qcom,memlat-cpu4/governor")

    # *** VIDC (encoder) config ***
    sudo_write("N", "/sys/kernel/debug/msm_vidc/clock_scaling")
    sudo_write("Y", "/sys/kernel/debug/msm_vidc/disable_thermal_mitigation")

  def configure_modem(self):
    sim_id = self.get_sim_info().get('sim_id', '')

    # configure modem as data-centric
    cmds = [
      'AT+QNVW=5280,0,"0102000000000000"',
      'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
      'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01',
    ]
    modem = self.get_modem()
    for cmd in cmds:
      try:
        modem.Command(cmd, math.ceil(TIMEOUT), dbus_interface=MM_MODEM, timeout=TIMEOUT)
      except Exception:
        pass

    # blue prime config
    if sim_id.startswith('8901410'):
      os.system('mmcli -m any --3gpp-set-initial-eps-bearer-settings="apn=Broadband"')

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

  def get_modem_data_usage(self):
    try:
      wwan = self.get_wwan()

      # Ensure refresh rate is set so values don't go stale
      refresh_rate = wwan.Get(NM_DEV_STATS, 'RefreshRateMs', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
      if refresh_rate != REFRESH_RATE_MS:
        u = type(refresh_rate)
        wwan.Set(NM_DEV_STATS, 'RefreshRateMs', u(REFRESH_RATE_MS), dbus_interface=DBUS_PROPS, timeout=TIMEOUT)

      tx = wwan.Get(NM_DEV_STATS, 'TxBytes', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
      rx = wwan.Get(NM_DEV_STATS, 'RxBytes', dbus_interface=DBUS_PROPS, timeout=TIMEOUT)
      return int(tx), int(rx)
    except Exception:
      return -1, -1

  def has_internal_panda(self):
    return True

  def reset_internal_panda(self):
    gpio_init(GPIO.STM_RST_N, True)

    gpio_set(GPIO.STM_RST_N, 1)
    time.sleep(2)
    gpio_set(GPIO.STM_RST_N, 0)

  def recover_internal_panda(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_init(GPIO.STM_BOOT0, True)

    gpio_set(GPIO.STM_RST_N, 1)
    gpio_set(GPIO.STM_BOOT0, 1)
    time.sleep(2)
    gpio_set(GPIO.STM_RST_N, 0)
    gpio_set(GPIO.STM_BOOT0, 0)


if __name__ == "__main__":
  t = Tici()
  t.initialize_hardware()
  t.set_power_save(False)
