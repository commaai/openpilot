#!/usr/bin/env python3

import time, mmap, sys, shutil, os, glob, subprocess
from tinygrad.helpers import to_mv, DEBUG, colored, ansilen
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.autogen.am import smu_v13_0_0
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.am.ip import AM_SOC21, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA

AM_VERSION = 0xA0000002

def bold(s): return f"\033[1m{s}\033[0m"

def color_temp(temp):
  if temp >= 87: return colored(f"{temp:>3}", "red")
  elif temp >= 80: return colored(f"{temp:>3}", "yellow")
  return f"{temp:>3}"

def color_voltage(voltage): return colored(f"{voltage/1000:>5.3f}V", "cyan")

def draw_bar(percentage, width=40, fill='█', empty='░'):
  filled_width = int(width * percentage)
  bar = fill * filled_width + empty * (width - filled_width)
  return f'[{bar}] {percentage*100:5.1f}%'

def same_line(strs:list[list[str]], split=8) -> list[str]:
  ret = []
  max_width_in_block = [max(ansilen(line) for line in block) for block in strs]
  max_height = max(len(block) for block in strs)
  for i in range(max_height):
    line = []
    for bid, block in enumerate(strs):
      if i < len(block): line.append(block[i] + ' ' * (split + max_width_in_block[bid] - ansilen(block[i])))
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
  def __init__(self, pcibus, vram_bar:memoryview, doorbell_bar:memoryview, mmio_bar:memoryview):
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
    self.soc21:AM_SOC21 = AM_SOC21(self)
    self.gmc:AM_GMC = AM_GMC(self)
    self.ih:AM_IH = AM_IH(self)
    self.psp:AM_PSP = AM_PSP(self)
    self.smu:AM_SMU = AM_SMU(self)

  def read_pci_state(self):
    with open(f"/sys/bus/pci/devices/{self.pcibus}/power_state", "r") as f: return f.read().strip().rstrip()

class SMICtx:
  def __init__(self):
    self.devs = []
    self.opened_pcidevs = []
    self.opened_pci_resources = {}
    self.prev_lines_cnt = 0

    remove_parts = ["Advanced Micro Devices, Inc. [AMD/ATI]", "VGA compatible controller:"]
    lspci = subprocess.check_output(["lspci"]).decode("utf-8").splitlines()
    self.lspci = {l.split()[0]: l.split(" ", 1)[1] for l in lspci}
    for k,v in self.lspci.items():
      for part in remove_parts: self.lspci[k] = self.lspci[k].replace(part, "").strip().rstrip()

  def _open_am_device(self, pcibus):
    if pcibus not in self.opened_pci_resources:
      bar_fds = {bar: os.open(f"/sys/bus/pci/devices/{pcibus}/resource{bar}", os.O_RDWR | os.O_SYNC) for bar in [0, 2, 5]}
      bar_size = {0: get_bar0_size(pcibus), 2: os.fstat(bar_fds[2]).st_size, 5: os.fstat(bar_fds[5]).st_size}

      def map_pci_range(bar):
        return to_mv(libc.mmap(0, bar_size[bar], mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, bar_fds[bar], 0), bar_size[bar])
      self.opened_pci_resources[pcibus] = (map_pci_range(0), None, map_pci_range(5).cast('I'))

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

  def draw(self):
    terminal_width, _ = shutil.get_terminal_size()

    dev_metrics = self.collect()
    dev_content = []
    for dev, metrics in dev_metrics.items():
      if dev.pci_state != "D0":
        dev_content.append([f"{colored('(sleep)', 'yellow')} {bold(dev.pcibus)}: {self.lspci[dev.pcibus[5:]]}"] +
                           [f"PCI State: {dev.pci_state}"] + [" "*107])
        continue

      device_line = [f"{bold(dev.pcibus)}: {self.lspci[dev.pcibus[5:]]}"] + [""]
      activity_line = [f"GFX Activity {draw_bar(metrics.SmuMetrics.AverageGfxActivity / 100, 50)}"] \
                    + [f"MEM Activity {draw_bar(metrics.SmuMetrics.AverageUclkActivity / 100, 50)}"] + [""]

      # draw_metrics_table(metrics, dev)
      temps_keys = [(k, name) for k, name in smu_v13_0_0.c__EA_TEMP_e__enumvalues.items()
                                if k < smu_v13_0_0.TEMP_COUNT and metrics.SmuMetrics.AvgTemperature[k] != 0]
      temps_table = ["=== Temps (C) ==="] + [f"{name:<15}: {color_temp(metrics.SmuMetrics.AvgTemperature[k])}" for k, name in temps_keys]

      voltage_keys = [(k, name) for k, name in smu_v13_0_0.c__EA_SVI_PLANE_e__enumvalues.items() if k < smu_v13_0_0.SVI_PLANE_COUNT]
      power_table = ["=== Power ==="] \
                  + [f"Fan Speed: {metrics.SmuMetrics.AvgFanRpm} RPM"] \
                  + [f"Fan Power: {metrics.SmuMetrics.AvgFanPwm}%"] \
                  + [f"Power: {metrics.SmuMetrics.AverageSocketPower:>3}W " +
                       draw_bar(metrics.SmuMetrics.AverageSocketPower / metrics.SmuMetrics.dGPU_W_MAX, 16)] \
                  + ["", "=== Voltages ==="] + [f"{name:<20}: {color_voltage(metrics.SmuMetrics.AvgVoltage[k])}" for k, name in voltage_keys]

      frequency_table = ["=== Frequencies ===",
        f"GFXCLK Target : {metrics.SmuMetrics.AverageGfxclkFrequencyTarget:>4} MHz",
        f"GFXCLK PreDs  : {metrics.SmuMetrics.AverageGfxclkFrequencyPreDs:>4} MHz",
        f"GFXCLK PostDs : {metrics.SmuMetrics.AverageGfxclkFrequencyPostDs:>4} MHz",
        f"FCLK PreDs    : {metrics.SmuMetrics.AverageFclkFrequencyPreDs:>4} MHz",
        f"FCLK PostDs   : {metrics.SmuMetrics.AverageFclkFrequencyPostDs:>4} MHz",
        f"MCLK PreDs    : {metrics.SmuMetrics.AverageMemclkFrequencyPreDs:>4} MHz",
        f"MCLK PostDs   : {metrics.SmuMetrics.AverageMemclkFrequencyPostDs:>4} MHz",
        f"VCLK0         : {metrics.SmuMetrics.AverageVclk0Frequency:>4} MHz",
        f"DCLK0         : {metrics.SmuMetrics.AverageDclk0Frequency:>4} MHz",
        f"VCLK1         : {metrics.SmuMetrics.AverageVclk1Frequency:>4} MHz",
        f"DCLK1         : {metrics.SmuMetrics.AverageDclk1Frequency:>4} MHz"]

      dev_content.append(device_line + activity_line + same_line([temps_table, power_table, frequency_table]))

    raw_text = 'AM Monitor'.center(terminal_width) + "\n" + "=" * terminal_width + "\n\n"
    for i in range(0, len(dev_content), 2):
      if i + 1 < len(dev_content): raw_text += '\n'.join(same_line([dev_content[i], dev_content[i+1]]))
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
