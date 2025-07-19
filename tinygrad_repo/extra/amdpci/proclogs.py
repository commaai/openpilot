import re, ctypes, sys

from tinygrad.runtime.autogen.am import am, mp_11_0, mp_13_0_0, nbio_4_3_0, mmhub_3_0_0, gc_11_0_0, osssys_6_0_0

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
  regs_offset = {13: {0: [3072, 37784576]}, 28: {0: [93184, 37754880], 1: [201327616, 201461760], 2: [209716224, 209850368], 3: [218104832, 218238976], 4: [226493440, 226627584], 5: [234882048, 235016192], 6: [243270656, 243404800]}, 21: {0: [28672, 12582912, 37795840, 130023424, 306184192], 1: [201326592, 201463808, 201465856, 204210176, 204472320], 2: [209715200, 209852416, 209854464, 212598784, 212860928], 3: [218103808, 218241024, 218243072, 220987392, 221249536], 4: [226492416, 226629632, 226631680, 229376000, 229638144], 5: [234881024, 235018240, 235020288, 237764608, 238026752], 6: [243269632, 243406848, 243408896, 246153216, 246415360]}, 22: {0: [18, 192, 13504, 36864, 37764096]}, 1: {0: [4704, 40960, 114688, 37760000]}, 2: {0: [3872, 37790720]}, 11: {0: [70656, 38103040]}, 12: {0: [106496, 37783552]}, 15: {0: [90112, 14417920, 14680064, 14942208, 38009856]}, 16: {0: [90112, 14417920, 14680064, 14942208, 38009856]}, 14: {0: [0, 20, 3360, 66560, 37859328, 67371008]}, 26: {0: [0, 20, 3360, 66560, 37859328, 67371008]}, 23: {0: [4256, 37789696]}, 33: {0: [0, 20, 3360, 66560, 37859328, 67371008]}, 25: {0: []}, 3: {0: [4704, 40960, 114688, 37760000]}, 4: {0: [4704, 40960, 114688, 37760000]}, 24: {0: [92160, 92672, 37752832, 54788096]}, 27: {0: [91648, 37751808], 1: [201339904, 201458176], 2: [209728512, 209846784], 3: [218117120, 218235392], 4: [226505728, 226624000], 5: [234894336, 235012608], 6: [243282944, 243401216]}, 29: {0: [201342976, 201344000, 205520896, 205537280], 1: [209731584, 209732608, 213909504, 213925888], 2: [218120192, 218121216, 222298112, 222314496], 3: [226508800, 226509824, 230686720, 230703104], 4: [234897408, 234898432, 239075328, 239091712], 5: [243286016, 243287040, 247463936, 247480320]}, 17: {0: [30720, 32256], 1: [31488, 73728]}}

  reg_names = {}
  def _prepare_registers(modules):
    for base, m in modules:
      for k, regval in m.__dict__.items():
        if k.startswith("reg") and not k.endswith("_BASE_IDX") and (base_idx:=getattr(m, f"{k}_BASE_IDX", None)) is not None:
          reg_names[regs_offset[am.__dict__.get(f"{base}_HWIP")][0][base_idx] + regval] = k

  _prepare_registers([("MP0", mp_13_0_0), ("NBIO", nbio_4_3_0), ("MMHUB", mmhub_3_0_0), ("GC", gc_11_0_0), ("OSSSYS", osssys_6_0_0)])

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