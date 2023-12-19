#!/usr/bin/env python3

if __name__ == "__main__":
  with open('../sensors/OS04CR4DEV217_AE02.ovd', 'rb') as f:
    dat = f.read()

  print(len(dat.splitlines()), "lines")

  cur = None
  configs = {}
  for line in dat.splitlines():
    line = line.split(b';')[0].strip()
    if line.startswith(b'@@ 0'):
      cur = line.strip().split(b' ')[-1]
      configs[cur] = list()
    elif line.startswith(b'6c') and len(line.split(b' ')) == 3:
      _, addr, val = line.split(b' ')
      configs[cur].append((addr, val))

  for name, regs in configs.items():
    with open(name.decode() + ".h", 'w') as f:
      f.write("""
#pragma once

const struct i2c_random_wr_payload start_reg_array_os04c10[] = {{0x100, 1}};
const struct i2c_random_wr_payload stop_reg_array_os04c10[] = {{0x100, 0}};

const struct i2c_random_wr_payload init_array_os04c10[] = {
      """.strip())
      f.write(f"\n  // {name.decode()}\n")
      for addr, val in regs:
        f.write("  {0x%s, 0x%s},\n" % (addr.decode(), val.decode()))
      f.write("};\n")
