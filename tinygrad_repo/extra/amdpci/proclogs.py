import re, ctypes, sys, importlib
from tinygrad.helpers import getenv

from tinygrad.runtime.support.am.amdev import AMDev, AMRegister

class GFXFake:
  def __init__(self): self.xccs = 8

class AMDFake(AMDev):
  def __init__(self, pci_dev, dma_regions=None):
    self.pci_dev, self.devfmt, self.dma_regions = pci_dev, pci_dev.pcibus, dma_regions
    self.vram, self.doorbell64, self.mmio = self.pci_dev.map_bar(0), self.pci_dev.map_bar(2, fmt='Q'), self.pci_dev.map_bar(5, fmt='I')
    self._run_discovery()
    self._build_regs()

    self.gfx = GFXFake()

amdev = importlib.import_module("tinygrad.runtime.support.am.amdev")
amdev.AMDev = AMDFake
from tinygrad.runtime.ops_amd import PCIIface

def parse_amdgpu_logs(log_content, register_names=None, *, only_xcc0: bool = False):
  register_map = register_names or {}

  def replace_register(match):
    reg = match.group(1)
    return f"Reading register {register_map.get(int(reg, 16), reg)}"

  processed_log = re.sub(r'Reading register (0x[0-9a-fA-F]+)', replace_register, log_content)

  def replace_register_2(match):
    reg = match.group(1)
    return f"Writing register {register_map.get(int(reg, 16), reg)}"

  processed_log = re.sub(r'Writing register (0x[0-9a-fA-F]+)', replace_register_2, processed_log)

  # remove timing prefix
  processed_log = re.sub(r'^\[\s*\d+(?:\.\d+)?\]\s*', '', processed_log, flags=re.MULTILINE)

  # keep only xcc=0 lines (but keep lines with no xcc at all)
  if only_xcc0:
    kept = []
    for line in processed_log.splitlines(True):
      if "xcc=" not in line or re.search(r'\bxcc=0\b', line): kept.append(line)
    processed_log = "".join(kept)

  return processed_log

def main():
  only_xcc0 = bool(getenv("ONLY_XCC0", 0))

  reg_names = {}
  dev = PCIIface(None, 0)
  for x, y in dev.dev_impl.__dict__.items():
    if isinstance(y, AMRegister):
      for xcc, addr in y.addr.items():
        reg_names[addr] = f"{x}, xcc={xcc}"

  with open(sys.argv[1], 'r') as f:
    log_content = f.read()

  processed_log = parse_amdgpu_logs(log_content, reg_names, only_xcc0=only_xcc0)

  with open(sys.argv[2], 'w') as f:
    f.write(processed_log)

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("Usage: <input_file_path> <output_file_path>")
    sys.exit(1)
  main()
