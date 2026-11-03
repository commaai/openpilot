import configparser
import glob
import json
import os
import socket
import subprocess
import time
from functools import cached_property, lru_cache
from pathlib import Path

from openpilot.cereal import log
from openpilot.common.esim.base import WEBBING_ICCID_PREFIX
from openpilot.common.utils import sudo_read, sudo_write
from openpilot.common.gpio import gpio_set, gpio_init, get_irqs_for_action
from openpilot.common.esim.base import LPABase
from openpilot.common.hardware.base import HardwareBase, ThermalConfig, ThermalZone
from openpilot.common.hardware.comma.pins import GPIO
from openpilot.common.hardware.comma.amplifier import Amplifier
from openpilot.common.kernel import upstream_kernel

MODEM_STATE_PATH = "/dev/shm/modem"

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


def affine_irq(val, action):
  irqs = get_irqs_for_action(action)
  if len(irqs) == 0:
    print(f"No IRQs found for '{action}'")
    return

  # power save takes the big cores offline; their IRQs stay where the kernel moved them until the cores come back
  # and the IRQs are affined again. the upstream kernel rejects an affinity with no online CPU
  online = Path(f"/sys/devices/system/cpu/cpu{val}/online")
  if online.exists() and online.read_text().strip() == "0":
    return

  for i in irqs:
    sudo_write(str(val), f"/proc/irq/{i}/smp_affinity_list")

_hwmon: dict[str, tuple[Path | None, float]] = {}
def hwmon(name: str) -> Path | None:
  """the hwmon device with this name, hwmon numbering depends on probe order. None until its driver has probed, or on
  devices without it (the comma four has no INA231): the scan costs ~3 ms, so a miss is only scanned again after 10 s"""
  path, scanned = _hwmon.get(name, (None, -10.))
  if path is None and time.monotonic() - scanned >= 10.:
    path = next((Path(p).parent for p in sorted(glob.glob("/sys/class/hwmon/hwmon*/name")) if Path(p).read_text().strip() == name), None)
    _hwmon[name] = (path, time.monotonic())
  return path

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

# the upstream kernel's IR LED, a leds-qcom-flash torch. it drives brightness b at b / LED_FULL of the DT's
# led-max-microamp, whatever max_brightness says
IR_TORCH = "/sys/class/leds/ir:torch"
IR_TORCH_MAX_MA = 300
LED_FULL = 255

def ir_torch_brightness(percent: int) -> int:
  """The ir:torch brightness that drives the IR LED with the current the downstream kernel drives for this percentage.
  Downstream, openpilot writes 0..300 mA to led:torch_2 and qpnp-flash-v2 rounds that to its 12.5 mA steps.
  Upstream, brightness b is floor(b * 300 / LED_FULL) mA, which leds-qcom-flash floors to its 5 mA steps."""
  ma = min(max(int(percent), 0), 100) * 300 // 100
  if ma == 0:
    return 0
  ua = max((ma * 1000 + 6250) // 12500, 1) * 12500
  return min(-(-ua * LED_FULL // (IR_TORCH_MAX_MA * 1000)), LED_FULL)

class MainlineThermalZone(ThermalZone):
  """a zone in the upstream kernel's DT. a missing one (a wrong name, or a sensor that didn't probe) reads 0 like on the
  downstream kernel, and is logged once"""
  missing_logged = False

  def read(self) -> float:
    temp = super().read()
    if self.zone_number < 0 and not self.missing_logged:
      from openpilot.common.swaglog import cloudlog
      cloudlog.error(f"no thermal zone named {self.name}")
      self.missing_logged = True
    return temp

def drm_fdinfo(pid: int) -> list[str]:
  """the fdinfo files of a process's DRM fds"""
  ret = []
  try:
    fds = os.listdir(f"/proc/{pid}/fd")
  except OSError:
    return ret
  for fd in fds:
    try:
      if os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/dri/"):
        ret.append(f"/proc/{pid}/fdinfo/{fd}")
    except OSError:
      pass
  return ret

def drm_engine_busy(fdinfo: str) -> dict[str, int]:
  """{drm-client-id: GPU busy ns} of a DRM fd, or {} if it's gone"""
  try:
    with open(fdinfo) as f:
      info = dict(line.split(":", 1) for line in f.read().splitlines() if ":" in line)
    return {info["drm-client-id"].strip(): int(info["drm-engine-gpu"].split()[0])}
  except (OSError, KeyError, ValueError, IndexError):
    return {}

def gpu_usage_percent(last: int, last_busy: dict[str, int], now: int, busy: dict[str, int]) -> float:
  """the GPU load between two samples of the clients' busy ns, counting a new client from its next sample"""
  if now <= last:
    return 0.
  return min(100., 100. * sum(b - last_busy.get(c, b) for c, b in busy.items()) / (now - last))

class HardwareComma(HardwareBase):
  """
    This platform covers the Snapdragon 845-based comma devices:
    - tici = comma three
    - tizi = comma 3X
    - mici = comma four

    We strictly use only the device codenames in this codebase for
    consistency, though all user-facing UI should use the product names.
  """

  def __init__(self):
    super().__init__()
    # GPU load on the upstream kernel: {pid: (when its fds were listed, its DRM fdinfo paths)} and the last
    # (time, {client: busy ns})
    self._drm_fdinfo: dict[int, tuple[int | None, list[str]]] = {}
    self._gpu_busy: tuple[int, dict[str, int]] = (time.monotonic_ns(), {})

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
    with open(f"{hwmon('ina231')}/in1_input") as f:
      return int(f.read())

  def get_current(self):
    with open(f"{hwmon('ina231')}/curr1_input") as f:
      return int(f.read())

  def set_ir_power(self, percent: int):
    if self.get_device_type() == "tizi":
      return

    if upstream_kernel():
      with open(f"{IR_TORCH}/brightness", "w") as f:
        f.write(f"{ir_torch_brightness(percent)}\n")
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
      if (iface := get_default_route_iface()):
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
    from openpilot.common.esim.lpa import LPA
    return LPA()

  def get_imei(self):
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
            network_strength = self.parse_strength(120 + max(-100, min(-20, dbm)))
      else:  # Cellular
        network_strength = self.parse_strength(self.get_modem_state().get('signal_quality', 0))
    except Exception:
      pass

    return network_strength

  def get_network_metered(self, network_type) -> bool:
    if network_type in (NetworkType.cell2G, NetworkType.cell3G, NetworkType.cell4G, NetworkType.cell5G):
      if self.get_modem_state().get("iccid", "").startswith(WEBBING_ICCID_PREFIX):
        return True
      from openpilot.common.params import Params
      return Params().get_bool("GsmMetered")
    try:
      if network_type == NetworkType.wifi:
        ssid = wpa_supplicant_cmd("STATUS").get("ssid", "")
        if ssid:
          # wpa_supplicant escapes non-printable bytes as \xNN; NM keyfile stores ASCII SSIDs as a literal and others as a byte;byte; list
          ssid_bytes = ssid.encode().decode('unicode_escape').encode('latin-1')
          ssid_keyfile_list = ';'.join(str(b) for b in ssid_bytes) + ';'

          nm_dirs = ("/run/NetworkManager/system-connections", "/data/etc/NetworkManager/system-connections")
          for fpath in (p for d in nm_dirs for p in Path(d).glob("*.nmconnection")):
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

  def get_modem_temperatures(self):
    return self.get_modem_state().get('temperatures', [])

  def get_current_power_draw(self):
    return (self.read_param_file(f"{hwmon('ina231')}/power1_input", int) / 1e6)

  def get_som_power_draw(self):
    if upstream_kernel():
      # in mV and mA
      som = hwmon("som_power")
      return self.read_param_file(f"{som}/in1_input", int) * self.read_param_file(f"{som}/curr1_input", int) / 1e6
    return (self.read_param_file("/sys/class/power_supply/bms/voltage_now", int) * self.read_param_file("/sys/class/power_supply/bms/current_now", int) / 1e12)

  def shutdown(self):
    subprocess.run("sudo poweroff", shell=True)

  def get_thermal_config(self):
    mici = self.get_device_type() == "mici"
    if upstream_kernel():
      # the DT's zone names, the board's have upstream's -thermal suffix too
      Zone = MainlineThermalZone
      return ThermalConfig(cpu=[Zone(f"cpu{i}-thermal") for i in range(8)],
                           gpu=[Zone("gpu-top-thermal"), Zone("gpu-bottom-thermal")],
                           dsp=Zone("q6-hvx-thermal"),
                           memory=Zone("mem-thermal"),
                           pmic=[Zone("pm8998-thermal")],
                           intake=Zone("intake-thermal") if mici else None,
                           exhaust=Zone("exhaust-thermal") if mici else None,
                           gnss=Zone("gnss-thermal") if mici else None,
                           bottomSoc=Zone("bottom-soc-thermal") if mici else None)
    intake, exhaust, gnss, bottomSoc = None, None, None, None
    if mici:
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

  @cached_property
  def _backlight(self):
    return "/sys/class/backlight/ae94000.dsi.0" if upstream_kernel() else "/sys/class/backlight/panel0-backlight"

  def set_display_power(self, on):
    try:
      with open(f"{self._backlight}/bl_power", "w") as f:
        f.write("0" if on else "4")
    except Exception:
      pass

  def set_screen_brightness(self, percentage):
    try:
      with open(f"{self._backlight}/max_brightness") as f:
        max_brightness = float(f.read().strip())

      val = int(percentage * (max_brightness / 100.))
      with open(f"{self._backlight}/brightness", "w") as f:
        f.write(str(val))
    except Exception:
      pass

  def get_screen_brightness(self):
    try:
      with open(f"{self._backlight}/max_brightness") as f:
        max_brightness = float(f.read().strip())

      with open(f"{self._backlight}/brightness") as f:
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
      if not powersave_enabled:
        # cap max core freq to 1689 Mhz
        sudo_write('1689600', f'/sys/devices/system/cpu/cpufreq/policy{n}/scaling_max_freq')

    # *** IRQ config ***

    # GPU, modeld core
    affine_irq(7, "gpu-irq" if upstream_kernel() else "kgsl-3d0")

    # camerad core
    camera_irqs = ("a5", "cci", "cpas_camnoc", "cpas-cdm", "csid", "ife", "csid-lite", "ife-lite")
    for n in camera_irqs:
      affine_irq(6, n)

  def get_gpu_usage_percent(self, pids=()):
    if upstream_kernel():
      return self._drm_gpu_usage_percent(pids)
    try:
      with open('/sys/class/kgsl/kgsl-3d0/gpubusy') as f:
        used, total = f.read().strip().split()
      return 100.0 * int(used) / int(total)
    except Exception:
      return 0

  def _drm_gpu_usage_percent(self, pids):
    # the GPU busy time of the given DRM clients, https://docs.kernel.org/gpu/drm-usage-stats.html
    now = time.monotonic_ns()
    busy = {}
    # listing a process's fds costs ~10 ms, so they're listed again every 10 s, for new DRM fds, or as soon as one of
    # them isn't a DRM fd any more (closed, or its number reused). dup'd fds are one client
    fdinfo = {}
    for pid in pids:
      listed, paths = self._drm_fdinfo.get(pid, (None, []))
      if listed is None or now - listed > 10e9:
        listed, paths = now, drm_fdinfo(pid)
      for path in paths:
        client = drm_engine_busy(path)
        if not client:
          listed = None
        busy.update(client)
      fdinfo[pid] = (listed, paths)
    self._drm_fdinfo = fdinfo
    last, last_busy = self._gpu_busy
    self._gpu_busy = (now, busy)
    return gpu_usage_percent(last, last_busy, now, busy)

  def initialize_hardware(self):
    if self.amplifier is not None:
      self.amplifier.initialize_configuration()

    # Allow hardwared to write engagement status to kmsg
    subprocess.run("sudo chmod a+w /dev/kmsg", shell=True)

    # Ensure fan gpio is enabled so fan runs until shutdown, also turned on at boot by the ABL
    gpio_init(GPIO.SOM_ST_IO, True)
    gpio_set(GPIO.SOM_ST_IO, True)

    # *** IRQ config ***

    if upstream_kernel():
      self._initialize_mainline()
    else:
      self._initialize_downstream()

    # pandad core
    affine_irq(3, "880000.spi" if upstream_kernel() else "spi_geni")  # SPI
    try:
      pid = subprocess.check_output(["pgrep", "-f", "spi0"], encoding='utf8').strip()
      subprocess.call(["sudo", "chrt", "-f", "-p", "1", pid])
      subprocess.call(["sudo", "taskset", "-pc", "3", pid])
    except subprocess.CalledProcessError as e:
      print(str(e))

  def _initialize_mainline(self):
    # move these off the default core, and off the cores AGNOS keeps for everything but openpilot
    affine_irq(2, "venus")  # encoders
    for i2c in ("890000.i2c", "894000.i2c", "a88000.i2c"):  # sensors, touch, amplifier
      affine_irq(2, i2c)
    # touch. the display IRQ is chained in the MDSS, so it can't be moved
    affine_irq(5, "ft3168" if self.get_device_type() == "mici" else "s6sy761_irq")

    # *** GPU config ***
    # 675 MHz like KGSL's pwrlevel 1. The GPU still runtime-suspends when idle, so offroad and after
    # openpilot exits it powers off and drops its DDR vote; onroad the models keep it awake
    sudo_write("675000000", "/sys/class/devfreq/5000000.gpu/max_freq")
    sudo_write("675000000", "/sys/class/devfreq/5000000.gpu/min_freq")

  def _initialize_downstream(self):
    # move these off the default core, and off the cores AGNOS keeps for everything but openpilot
    affine_irq(2, "msm_vidc")  # encoders
    affine_irq(2, "i2c_geni")  # sensors

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

  def get_modem_data_usage(self):
    ms = self.get_modem_state()
    return ms.get('tx_bytes', -1), ms.get('rx_bytes', -1)

  def reset_internal_panda(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_init(GPIO.STM_BOOT0, True)

    gpio_set(GPIO.STM_RST_N, True)
    gpio_set(GPIO.STM_BOOT0, False)
    time.sleep(0.01)
    gpio_set(GPIO.STM_RST_N, False)

  def recover_internal_panda(self):
    gpio_init(GPIO.STM_RST_N, True)
    gpio_init(GPIO.STM_BOOT0, True)

    gpio_set(GPIO.STM_RST_N, True)
    gpio_set(GPIO.STM_BOOT0, True)
    time.sleep(0.01)
    gpio_set(GPIO.STM_RST_N, False)
    time.sleep(0.01)
    gpio_set(GPIO.STM_BOOT0, False)

  def booted(self):
    # the encoder normally boots within 8s, but on rare occasions takes 30+s
    if upstream_kernel():
      # Venus registers its video nodes once its firmware has booted
      encoder_booted = any(Path(p).read_text().strip() == "qcom-venus-encoder" for p in glob.glob("/sys/class/video4linux/*/name"))
    else:
      encoder_booted = "Core state: 0" not in sudo_read("/sys/kernel/debug/msm_vidc/core0/info")
    return encoder_booted or time.monotonic() >= 60*2

if __name__ == "__main__":
  t = HardwareComma()
  t.initialize_hardware()
  t.set_power_save(False)
  print(t.get_sim_info())
