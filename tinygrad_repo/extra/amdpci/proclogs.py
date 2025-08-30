import re, ctypes, sys, importlib

from tinygrad.runtime.support.am.amdev import AMDev, AMRegister
class AMDFake(AMDev):
  def __init__(self, devfmt, vram, doorbell, mmio, dma_regions=None):
    self.devfmt, self.vram, self.doorbell64, self.mmio, self.dma_regions = devfmt, vram, doorbell, mmio, dma_regions
    self._run_discovery()
    self._build_regs()

amdev = importlib.import_module("tinygrad.runtime.support.am.amdev")
amdev.AMDev = AMDFake

from tinygrad.runtime.ops_amd import PCIIface

def parse_amdgpu_logs(log_content, register_names=None):
  register_map = register_names

  final = ""
  def replace_register(match):
    register = match.group(1)
    return f"Reading register {register_map.get(int(register, base=16), register)}"

  pattern = r'Reading register (0x[0-9a-fA-F]+)'

  processed_log = re.sub(pattern, replace_register, log_content)

  def replace_register_2(match):
    register = match.group(1)
    return f"Writing register {register_map.get(int(register, base=16), register)}"

  pattern = r'Writing register (0x[0-9a-fA-F]+)'
  processed_log = re.sub(pattern, replace_register_2, processed_log)
  return processed_log

def main():
  reg_names = {}
  dev = PCIIface(None, 0)
  for x, y in dev.dev_impl.__dict__.items():
    if isinstance(y, AMRegister):
      for inst, addr in y.addr.keys(): reg_names[addr] = f"{x}, xcc={inst}"

  with open(sys.argv[1], 'r') as f:
    log_content = log_content_them = f.read()

  processed_log = parse_amdgpu_logs(log_content, reg_names)

  with open(sys.argv[2], 'w') as f:
    f.write(processed_log)

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("Usage: <input_file_path> <output_file_path>")
    sys.exit(1)

  main()