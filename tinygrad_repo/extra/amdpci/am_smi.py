#!/usr/bin/env python3

import time, mmap, sys, shutil, os, glob, subprocess
from tinygrad.helpers import DEBUG, colored, ansilen
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.am.ip import AM_SOC, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA

AM_VERSION = 0xA0000003

def bold(s): return f"\033[1m{s}\033[0m"

def trim(s:str, length:int) -> str:
  if len(s) > length: return s[:length-3] + "..."
  return s

def pad(x:str, length:int) -> str:
  if len(x) < length: return x + " " * (length - len(x))
  return x

def color_temp(temp):
  if temp >= 87: return colored(f"{temp:>3}", "red")
  elif temp >= 80: return colored(f"{temp:>3}", "yellow")
  return f"{temp:>3}"

def color_voltage(voltage): return colored(f"{voltage/1000:>5.3f}V", "cyan")

def draw_bar(percentage, width=40, fill='█', empty='░'):
  filled_width = int(width * percentage)
  bar = fill * filled_width + empty * (width - filled_width)
  return f'[{bar}] {percentage*100:5.1f}%'

def same_line(strs:list[list[str]|None], split=8) -> list[str]:
  strs = [s for s in strs if s is not None]
  if len(strs) == 0: return []

  ret = []
  max_width_in_block = [max(ansilen(line) for line in block) for block in strs]
  max_height = max(len(block) for block in strs)
  for i in range(max_height):
    line = []
    for bid, block in enumerate(strs):
      if i < len(block): line.append(block[i] + (' ' * (split + max_width_in_block[bid] - ansilen(block[i])) if bid != len(strs) - 1 else ''))
      else: line.append(' ' * (split + max_width_in_block[bid]))
    ret.append(' '.join(line))
  return ret

def get_bar0_size(pcibus):
  resource_file = f"/sys/bus/pci/devices/{pcibus}/resource"
  if not os.path.exists(resource_file): raise FileNotFoundError(f"Resource file not found: {resource_file}")

  with open(resource_file, "r") as f: lines = f.readlines()
  bar0_info = lines[0].split()
  if len(bar0_info) < 3: raise ValueError("Unexpected resource file format for BAR0.")

  start_hex, end_hex, _flags = bar0_info
  return int(end_hex, 16) - int(start_hex, 16) + 1

class AMSMI(AMDev):
  def __init__(self, pcibus, vram_bar:MMIOInterface, doorbell_bar:MMIOInterface, mmio_bar:MMIOInterface):
    self.pcibus = pcibus
    self.vram, self.doorbell64, self.mmio = vram_bar, doorbell_bar, mmio_bar
    self.pci_state = self.read_pci_state()
    if self.pci_state == "D0": self._init_from_d0()

  def _init_from_d0(self):
    self._run_discovery()
    self._build_regs()

    if self.reg("regSCRATCH_REG7").read() != AM_VERSION:
      raise Exception(f"Unsupported AM version: {self.reg('regSCRATCH_REG7').read():x}")

    self.is_booting, self.smi_dev = True, True
    self.partial_boot = True # do not init anything
    self.mm = AMMemoryManager(self, self.vram_size)

    # Initialize IP blocks
    self.soc:AM_SOC = AM_SOC(self)
    self.gmc:AM_GMC = AM_GMC(self)
    self.ih:AM_IH = AM_IH(self)
    self.psp:AM_PSP = AM_PSP(self)
    self.smu:AM_SMU = AM_SMU(self)

    for ip in [self.soc, self.gmc, self.ih, self.psp, self.smu]: ip.init_sw()

  def read_pci_state(self):
    with open(f"/sys/bus/pci/devices/{self.pcibus}/power_state", "r") as f: return f.read().strip().rstrip()

class SMICtx:
  def __init__(self):
    self.devs = []
    self.opened_pcidevs = []
    self.opened_pci_resources = {}
    self.prev_lines_cnt = 0
    self.prev_terminal_width = 0

    remove_parts = ["Advanced Micro Devices, Inc. [AMD/ATI]", "VGA compatible controller:"]
    lspci = subprocess.check_output(["lspci"]).decode("utf-8").splitlines()
    self.lspci = {l.split()[0]: l.split(" ", 1)[1] for l in lspci}
    for k,v in self.lspci.items():
      for part in remove_parts: self.lspci[k] = self.lspci[k].replace(part, "").strip().rstrip()

  def _open_am_device(self, pcibus):
    if pcibus not in self.opened_pci_resources:
      bar_fds = {bar: os.open(f"/sys/bus/pci/devices/{pcibus}/resource{bar}", os.O_RDWR | os.O_SYNC) for bar in [0, 2, 5]}
      bar_size = {0: get_bar0_size(pcibus), 2: os.fstat(bar_fds[2]).st_size, 5: os.fstat(bar_fds[5]).st_size}

      def map_pci_range(bar, fmt='B'):
        return MMIOInterface(libc.mmap(0, bar_size[bar], mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, bar_fds[bar], 0), bar_size[bar], fmt)
      self.opened_pci_resources[pcibus] = (map_pci_range(0), None, map_pci_range(5, 'I'))

    try:
      self.devs.append(AMSMI(pcibus, *self.opened_pci_resources[pcibus]))
    except Exception as e:
      if DEBUG >= 2: print(f"Failed to open AM device {pcibus}: {e}")
      return

    self.opened_pcidevs.append(pcibus)
    if DEBUG >= 2: print(f"Opened AM device {pcibus}")

  def rescan_devs(self):
    pattern = os.path.join('/tmp', 'am_*.lock')
    for d in [f[8:-5] for f in glob.glob(pattern)]:
      if d not in self.opened_pcidevs:
        self._open_am_device(d)

    for d in self.devs:
      if d.read_pci_state() != d.pci_state:
        d.pci_state = d.read_pci_state()
        if d.pci_state == "D0": d._init_from_d0()
        os.system('clear')

      if d.pci_state == "D0" and d.reg("regSCRATCH_REG7").read() != AM_VERSION:
        self.devs.remove(d)
        self.opened_pcidevs.remove(d.pcibus)
        os.system('clear')
        if DEBUG >= 2: print(f"Removed AM device {d.pcibus}")

  def collect(self): return {d: d.smu.read_metrics() if d.pci_state == "D0" else None for d in self.devs}

  def get_gfx_activity(self, dev, metrics): return metrics.SmuMetrics.AverageGfxActivity
  def get_mem_activity(self, dev, metrics): return metrics.SmuMetrics.AverageUclkActivity

  def get_temps(self, dev, metrics, compact=False):
    temps_keys = [(k, name) for k, name in dev.smu.smu_mod.c__EA_TEMP_e__enumvalues.items()
                  if k < dev.smu.smu_mod.TEMP_COUNT and metrics.SmuMetrics.AvgTemperature[k] != 0]
    if compact: temps_keys = [(k, name) for k, name in temps_keys if k in (dev.smu.smu_mod.TEMP_HOTSPOT, dev.smu.smu_mod.TEMP_MEM)]
    return {name: metrics.SmuMetrics.AvgTemperature[k] for k, name in temps_keys}

  def get_voltage(self, dev, metrics, compact=False):
    voltage_keys = [(k, name) for k, name in dev.smu.smu_mod.c__EA_SVI_PLANE_e__enumvalues.items()
                        if k < dev.smu.smu_mod.SVI_PLANE_COUNT and metrics.SmuMetrics.AvgVoltage[k] != 0]
    return {name: metrics.SmuMetrics.AvgVoltage[k] for k, name in voltage_keys}

  def get_busy_threshold(self, dev):
    match dev.ip_ver[am.MP1_HWIP]:
      case (14, 0, 2): return 5
      case _: return 15

  def get_gfx_freq(self, dev, metrics):
    return metrics.SmuMetrics.AverageGfxclkFrequencyPostDs if self.get_gfx_activity(dev, metrics) <= self.get_busy_threshold(dev) else \
          metrics.SmuMetrics.AverageGfxclkFrequencyPreDs

  def get_mem_freq(self, dev, metrics):
    return metrics.SmuMetrics.AverageMemclkFrequencyPostDs if self.get_mem_activity(dev, metrics) <= self.get_busy_threshold(dev) else \
           metrics.SmuMetrics.AverageMemclkFrequencyPreDs

  def get_fckl_freq(self, dev, metrics):
    return metrics.SmuMetrics.AverageFclkFrequencyPostDs if self.get_mem_activity(dev, metrics) <= self.get_busy_threshold(dev) else \
           metrics.SmuMetrics.AverageFclkFrequencyPreDs

  def get_fan_rpm_pwm(self, dev, metrics): return metrics.SmuMetrics.AvgFanRpm, metrics.SmuMetrics.AvgFanPwm

  def get_power(self, dev, metrics): return metrics.SmuMetrics.AverageSocketPower, metrics.SmuMetrics.dGPU_W_MAX

  def draw(self):
    terminal_width, terminal_height = shutil.get_terminal_size()
    if self.prev_terminal_width != terminal_width or self.prev_terminal_height != terminal_height:
      os.system('clear')
    self.prev_terminal_width, self.prev_terminal_height = terminal_width, terminal_height

    padding = 8
    col_size = (terminal_width) // 2 - padding
    activity_line_width = 50 if terminal_width > 170 else \
                         (30 if terminal_width > 130 else \
                         (16 if terminal_width > 92 else \
                         max(0, terminal_width - 77)))

    dev_metrics = self.collect()
    dev_content = []
    for dev, metrics in dev_metrics.items():
      if dev.pci_state != "D0":
        dev_content.append([f"{colored('(sleep)', 'yellow')} {bold(dev.pcibus)}: {self.lspci[dev.pcibus[5:]]}"] +
                           [pad(f"PCI State: {dev.pci_state}", col_size)])
        continue

      device_line = [f"{bold(dev.pcibus)} {trim(self.lspci[dev.pcibus[5:]], col_size - 20)}"] + [pad("", col_size)]
      activity_line = [f"GFX Activity {draw_bar(self.get_gfx_activity(dev, metrics) / 100, activity_line_width)}"] \
                    + [f"MEM Activity {draw_bar(self.get_mem_activity(dev, metrics) / 100, activity_line_width)}"]

      temps_data, temps_data_compact = self.get_temps(dev, metrics), self.get_temps(dev, metrics, compact=True)
      temps_table = ["=== Temps (°C) ==="] + [f"{name:<16}: {color_temp(val)}" for name, val in temps_data.items()]
      temps_table_compact = ["Temps (°C):" + '/'.join([f"{color_temp(val)} {name}" for name, val in temps_data_compact.items()])]

      fan_rpm, fan_pwm = self.get_fan_rpm_pwm(dev, metrics)
      power_table = ["=== Power ==="] + [f"Fan Speed: {fan_rpm} RPM"] + [f"Fan Power: {fan_pwm}%"]

      total_power, max_power = self.get_power(dev, metrics)
      power_line = [f"Power: {total_power:>3}W " + draw_bar(total_power / max_power, 16)]
      power_line_compact = [f"Power:  {total_power:>3}W " + draw_bar(total_power / max_power, activity_line_width)]

      voltage_data = self.get_voltage(dev, metrics)
      voltage_table = ["=== Voltages ==="] + [f"{name:<20}: {color_voltage(voltage)}" for name, voltage in voltage_data.items()]

      gfx_freq = self.get_gfx_freq(dev, metrics)
      mclk_freq = self.get_mem_freq(dev, metrics)
      fclk_freq = self.get_fckl_freq(dev, metrics)

      frequency_table = ["=== Frequencies ===", f"GFXCLK: {gfx_freq:>4} MHz", f"FCLK  : {fclk_freq:>4} MHz", f"MCLK  : {mclk_freq:>4} MHz"]

      if self.prev_terminal_width >= 231:
        power_table += power_line + [""] + voltage_table
        activity_line += [""]
      elif self.prev_terminal_width >= 171:
        power_table += power_line + [""] + frequency_table
        activity_line += [""]
        frequency_table = None
      elif self.prev_terminal_width >= 121:
        temps_table = None
        activity_line += power_line_compact
      else:
        temps_table = None
        power_table = None
        frequency_table = None
        activity_line += power_line_compact

      dev_content.append(device_line + activity_line + same_line([temps_table, power_table, frequency_table]))

    raw_text = 'AM Monitor'.center(terminal_width) + "\n" + "=" * terminal_width + "\n\n"
    for i in range(0, len(dev_content), 2):
      if i + 1 < len(dev_content): raw_text += '\n'.join(same_line([dev_content[i], dev_content[i+1]], split=padding))
      else: raw_text += '\n'.join(dev_content[i])
      if i + 2 < len(dev_content): raw_text += "\n" + "=" * terminal_width + "\n\n"

    sys.stdout.write(f'\033[{self.prev_lines_cnt}A')
    sys.stdout.flush()
    print(raw_text)

    self.prev_lines_cnt = len(raw_text.splitlines()) + 2

if __name__ == "__main__":
  try:
    os.system('clear')
    smi_ctx = SMICtx()
    while True:
      smi_ctx.rescan_devs()
      smi_ctx.draw()
      time.sleep(1)
  except KeyboardInterrupt: print("Exiting...")
