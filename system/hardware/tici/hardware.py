import configparser
import json
import os
import socket
import subprocess
import time
from functools import cached_property, lru_cache
from pathlib import Path

from cereal import log
from openpilot.common.utils import sudo_read, sudo_write
from openpilot.common.gpio import gpio_set, gpio_init, get_irqs_for_action
from openpilot.system.hardware.base import HardwareBase, LPABase, ThermalConfig, ThermalZone
from openpilot.system.hardware.tici import iwlist
from openpilot.system.hardware.tici.lpa import TiciLPA
from openpilot.system.hardware.tici.pins import GPIO
from openpilot.system.hardware.tici.amplifier import Amplifier

NM_CONNECTIONS_DIRS = ("/run/NetworkManager/system-connections", "/data/etc/NetworkManager/system-connections")
MODEM_STATE_PATH = "/dev/shm/modem"

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


def affine_irq(val, action):
  irqs = get_irqs_for_action(action)
  if len(irqs) == 0:
    print(f"No IRQs found for '{action}'")
    return

  for i in irqs:
    sudo_write(str(val), f"/proc/irq/{i}/smp_affinity_list")

@lru_cache
def get_device_type():
  # lru_cache and cache can cause memory leaks when used in classes
  with open("/sys/firmware/devicetree/base/model") as f:
    model = f.read().strip('\x00')
  return model.split('comma ')[-1]

def wpa_supplicant_cmd(cmd: str, timeout: float = 0.2) -> dict[str, str]:
  with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
    sock.settimeout(timeout)
    sock.bind(f"\0openpilot-wpa-{os.getpid()}-{time.monotonic_ns()}")
    sock.connect("/run/wpa_supplicant/wlan0")
    sock.send(cmd.encode())

    while True:
      out = sock.recv(8192).decode("utf-8", "replace")
      if out.startswith("<"):
        continue
      if out.startswith("FAIL"):
        return {}
      return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)

def get_default_route_iface():
  with open("/proc/net/route") as f:
    routes = [(int(route[6]), route[0]) for line in f.readlines()[1:] if (route := line.split())[1] == "00000000" and int(route[3], 16) & 0x1]
  return min(routes)[1] if routes else None

class Tici(HardwareBase):
  @cached_property
  def amplifier(self):
    if self.get_device_type() == "mici":
      return None
    return Amplifier()

  def get_modem_state(self) -> dict:
    try:
      with open(MODEM_STATE_PATH) as f:
        return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
      return {}

  def get_os_version(self):
    with open("/VERSION") as f:
      return f.read().strip()

  def get_device_type(self):
    return get_device_type()

  def reboot(self, reason=None):
    subprocess.check_output(["sudo", "reboot"])

  def uninstall(self):
    Path("/data/__system_reset__").touch()
    os.sync()
    self.reboot()

  def get_serial(self):
    return self.get_cmdline()['androidboot.serialno']

  def get_voltage(self):
    with open("/sys/class/hwmon/hwmon1/in1_input") as f:
      return int(f.read())

  def get_current(self):
    with open("/sys/class/hwmon/hwmon1/curr1_input") as f:
      return int(f.read())

  def set_ir_power(self, percent: int):
    if self.get_device_type() == "tizi":
      return

    value = int((percent / 100) * 300)
    with open("/sys/class/leds/led:switch_2/brightness", "w") as f:
      f.write("0\n")
    with open("/sys/class/leds/led:torch_2/brightness", "w") as f:
      f.write(f"{value}\n")
    with open("/sys/class/leds/led:switch_2/brightness", "w") as f:
      f.write(f"{value}\n")

  def get_network_type(self):
    try:
      iface = get_default_route_iface()
      if iface.startswith('wlan'):
        return NetworkType.wifi
      if iface.startswith('eth'):
        return NetworkType.ethernet
    except Exception:
      pass

    ms = self.get_modem_state()
    if ms.get('connected'):
      nt = ms.get('network_type', '')
      if nt == 'nr':
        return NetworkType.cell5G
      elif nt == 'lte':
        return NetworkType.cell4G
      elif nt in ('utran', 'umts'):
        return NetworkType.cell3G
      elif nt == 'gsm':
        return NetworkType.cell2G
    return NetworkType.none

  def get_sim_info(self):
    ms = self.get_modem_state()
    sim_id = ms.get('iccid', '')
    return {
      'sim_id': sim_id,
      'mcc_mnc': ms.get('mcc_mnc') or None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"] if not sim_id else ["READY"],
      'data_connected': ms.get('connected', False),
    }

  def get_sim_lpa(self) -> LPABase:
    return TiciLPA()

  def get_imei(self, slot):
    if slot != 0:
      return ""
    return self.get_modem_state().get('imei', '')

  def get_network_info(self):
    if self.get_device_type() == "mici":
      return None

    ms = self.get_modem_state()
    return {
      'technology': ms.get('network_type', '').upper() if ms.get('network_type') else '',
      'operator': ms.get('operator', ''),
      'band': ms.get('band', ''),
      'channel': ms.get('channel', 0),
      'extra': ms.get('extra', ''),
      'state': ms.get('state', 'UNKNOWN'),
    }

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
      elif network_type == NetworkType.ethernet:
        network_strength = NetworkStrength.great
      elif network_type == NetworkType.wifi:
        rssi = wpa_supplicant_cmd("SIGNAL_POLL").get("RSSI")
        if rssi is not None:
          dbm = int(rssi)
          if -100 < dbm <= 0:
            network_strength = self.parse_strength(120 + max(-90, min(-20, dbm)))
      else:  # Cellular
        network_strength = self.parse_strength(self.get_modem_state().get('signal_quality', 0))
    except Exception:
      pass

    return network_strength

  def get_network_metered(self, network_type) -> bool:
    if network_type in (NetworkType.cell2G, NetworkType.cell3G, NetworkType.cell4G, NetworkType.cell5G):
      from openpilot.common.params import Params
      return Params().get_bool("GsmMetered")
    try:
      if network_type == NetworkType.wifi:
        ssid = wpa_supplicant_cmd("STATUS").get("ssid", "")
        if ssid:
          # wpa_supplicant escapes non-printable bytes as \xNN; NM keyfile stores ASCII SSIDs as a literal and others as a byte;byte; list
          ssid_bytes = ssid.encode().decode('unicode_escape').encode('latin-1')
          ssid_keyfile_list = ';'.join(str(b) for b in ssid_bytes) + ';'

          for fpath in (p for d in NM_CONNECTIONS_DIRS for p in Path(d).glob("*.nmconnection")):
            raw = sudo_read(str(fpath))
            if not raw:
              continue
            cp = configparser.ConfigParser(interpolation=None)
            try:
              cp.read_string(raw)
              keyfile_ssid = cp.get("wifi", "ssid", fallback="")
              if keyfile_ssid != ssid and keyfile_ssid != ssid_keyfile_list:
                continue
              metered = cp.getint("connection", "metered", fallback=0)
            except (configparser.Error, ValueError):
              continue
            if metered == 1:  # NM_METERED_YES
              return True
            if metered == 2:  # NM_METERED_NO
              return False
            break
    except Exception:
      pass

    return super().get_network_metered(network_type)

  def get_modem_version(self):
    return self.get_modem_state().get('modem_version') or None

  def get_modem_temperatures(self):
    return self.get_modem_state().get('temperatures', [])

  def get_current_power_draw(self):
    return (self.read_param_file("/sys/class/hwmon/hwmon1/power1_input", int) / 1e6)

  def get_som_power_draw(self):
    return (self.read_param_file("/sys/class/power_supply/bms/voltage_now", int) * self.read_param_file("/sys/class/power_supply/bms/current_now", int) / 1e12)

  def shutdown(self):
    os.system("sudo poweroff")

  def get_thermal_config(self):
    intake, exhaust, gnss, bottomSoc = None, None, None, None
    if self.get_device_type() == "mici":
      gnss = ThermalZone("gnss")
      intake = ThermalZone("intake")
      exhaust = ThermalZone("exhaust")
      bottomSoc = ThermalZone("bottom_soc")
    return ThermalConfig(cpu=[ThermalZone(f"cpu{i}-silver-usr") for i in range(4)] +
                             [ThermalZone(f"cpu{i}-gold-usr") for i in range(4)],
                         gpu=[ThermalZone("gpu0-usr"), ThermalZone("gpu1-usr")],
                         dsp=ThermalZone("compute-hvx-usr"),
                         memory=ThermalZone("ddr-usr"),
                         pmic=[ThermalZone("pm8998_tz"), ThermalZone("pm8005_tz")],
                         intake=intake,
                         exhaust=exhaust,
                         gnss=gnss,
                         bottomSoc=bottomSoc)

  def set_display_power(self, on):
    try:
      with open("/sys/class/backlight/panel0-backlight/bl_power", "w") as f:
        f.write("0" if on else "4")
    except Exception:
      pass

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
    if self.amplifier is not None:
      self.amplifier.set_global_shutdown(amp_disabled=powersave_enabled)
      if not powersave_enabled:
        self.amplifier.initialize_configuration()

    # *** CPU config ***

    # offline big cluster
    for i in range(4, 8):
      val = '0' if powersave_enabled else '1'
      sudo_write(val, f'/sys/devices/system/cpu/cpu{i}/online')

    for n in ('0', '4'):
      if powersave_enabled and n == '4':
        continue
      gov = 'ondemand' if powersave_enabled else 'performance'
      sudo_write(gov, f'/sys/devices/system/cpu/cpufreq/policy{n}/scaling_governor')

    # *** IRQ config ***

    # GPU, modeld core
    affine_irq(7, "kgsl-3d0")

    # camerad core
    camera_irqs = ("a5", "cci", "cpas_camnoc", "cpas-cdm", "csid", "ife", "csid-lite", "ife-lite")
    for n in camera_irqs:
      affine_irq(6, n)

  def get_gpu_usage_percent(self):
    try:
      with open('/sys/class/kgsl/kgsl-3d0/gpubusy') as f:
        used, total = f.read().strip().split()
      return 100.0 * int(used) / int(total)
    except Exception:
      return 0

  def initialize_hardware(self):
    if self.amplifier is not None:
      self.amplifier.initialize_configuration()

    # Allow hardwared to write engagement status to kmsg
    os.system("sudo chmod a+w /dev/kmsg")

    # Ensure fan gpio is enabled so fan runs until shutdown, also turned on at boot by the ABL
    gpio_init(GPIO.SOM_ST_IO, True)
    gpio_set(GPIO.SOM_ST_IO, 1)

    # *** IRQ config ***

    # mask off big cluster from default affinity
    sudo_write("f", "/proc/irq/default_smp_affinity")

    # move these off the default core
    affine_irq(1, "msm_vidc")  # encoders
    affine_irq(1, "i2c_geni")  # sensors

    # *** GPU config ***
    # https://github.com/commaai/agnos-kernel-sdm845/blob/master/arch/arm64/boot/dts/qcom/sdm845-gpu.dtsi#L216
    affine_irq(5, "fts_ts")    # touch
    affine_irq(5, "msm_drm")   # display
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/min_pwrlevel")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/max_pwrlevel")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/force_bus_on")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/force_clk_on")
    sudo_write("1", "/sys/class/kgsl/kgsl-3d0/force_rail_on")
    sudo_write("1000", "/sys/class/kgsl/kgsl-3d0/idle_timer")
    sudo_write("performance", "/sys/class/kgsl/kgsl-3d0/devfreq/governor")
    sudo_write("710", "/sys/class/kgsl/kgsl-3d0/max_clock_mhz")

    # setup governors
    sudo_write("performance", "/sys/class/devfreq/soc:qcom,cpubw/governor")
    sudo_write("performance", "/sys/class/devfreq/soc:qcom,memlat-cpu0/governor")
    sudo_write("performance", "/sys/class/devfreq/soc:qcom,memlat-cpu4/governor")

    # *** VIDC (encoder) config ***
    sudo_write("N", "/sys/kernel/debug/msm_vidc/clock_scaling")
    sudo_write("Y", "/sys/kernel/debug/msm_vidc/disable_thermal_mitigation")

    # pandad core
    affine_irq(3, "spi_geni")         # SPI
    try:
      pid = subprocess.check_output(["pgrep", "-f", "spi0"], encoding='utf8').strip()
      subprocess.call(["sudo", "chrt", "-f", "-p", "1", pid])
      subprocess.call(["sudo", "taskset", "-pc", "3", pid])
    except subprocess.CalledProcessException as e:
      print(str(e))

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
    ms = self.get_modem_state()
    return ms.get('tx_bytes', -1), ms.get('rx_bytes', -1)

  def has_internal_panda(self):
    return True

  def reset_internal_panda(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_init(GPIO.STM_BOOT0, True)

    gpio_set(GPIO.STM_RST_N, 1)
    gpio_set(GPIO.STM_BOOT0, 0)
    time.sleep(0.01)
    gpio_set(GPIO.STM_RST_N, 0)

  def recover_internal_panda(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_init(GPIO.STM_BOOT0, True)

    gpio_set(GPIO.STM_RST_N, 1)
    gpio_set(GPIO.STM_BOOT0, 1)
    time.sleep(0.01)
    gpio_set(GPIO.STM_RST_N, 0)
    time.sleep(0.01)
    gpio_set(GPIO.STM_BOOT0, 0)

  def booted(self):
    # this normally boots within 8s, but on rare occasions takes 30+s
    encoder_state = sudo_read("/sys/kernel/debug/msm_vidc/core0/info")
    if "Core state: 0" in encoder_state and (time.monotonic() < 60*2):
      return False
    return True

if __name__ == "__main__":
  t = Tici()
  t.initialize_hardware()
  t.set_power_save(False)
  print(t.get_sim_info())
