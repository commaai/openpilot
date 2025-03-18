#!/usr/bin/env python3
import subprocess
from collections import defaultdict


def check_space(file, mcu):
  MCUS = {
    "H7": {
      ".flash": 1024*1024, # FLASH
      ".dtcmram": 128*1024, # DTCMRAM
      ".itcmram": 64*1024, # ITCMRAM
      ".axisram": 320*1024, # AXI SRAM
      ".sram12": 32*1024, # SRAM1(16kb) + SRAM2(16kb)
      ".sram4": 16*1024, # SRAM4
      ".backup_sram": 4*1024, # SRAM4
    },
    "F4": {
      ".flash": 1024*1024, # FLASH
      ".dtcmram": 256*1024, # RAM
      ".ram_d1": 64*1024, # RAM2
    },
  }
  IGNORE_LIST = [
    ".ARM.attributes",
    ".comment",
    ".debug_line",
    ".debug_info",
    ".debug_abbrev",
    ".debug_aranges",
    ".debug_str",
    ".debug_ranges",
    ".debug_loc",
    ".debug_frame",
    ".debug_line_str",
    ".debug_rnglists",
    ".debug_loclists",
  ]
  FLASH = [
    ".isr_vector",
    ".text",
    ".rodata",
    ".data"
  ]
  RAM = [
    ".data",
    ".bss",
    "._user_heap_stack" # _user_heap_stack considered free?
  ]

  result = {}
  calcs = defaultdict(int)

  output = str(subprocess.check_output(f"arm-none-eabi-size -x --format=sysv {file}", shell=True), 'utf-8')

  for row in output.split('\n'):
    pop = False
    line = row.split()
    if len(line) == 3 and line[0].startswith('.'):
      if line[0] in IGNORE_LIST:
        continue
      result[line[0]] = [line[1], line[2]]
      if line[0] in FLASH:
        calcs[".flash"] += int(line[1], 16)
        pop = True
      if line[0] in RAM:
        calcs[".dtcmram"] += int(line[1], 16)
        pop = True
      if pop:
        result.pop(line[0])

  if len(result):
    for line in result:
      calcs[line] += int(result[line][0], 16)

  print(f"=======SUMMARY FOR {mcu} FILE {file}=======")
  for line in calcs:
    if line in MCUS[mcu]:
      used_percent = (100 - (MCUS[mcu][line] - calcs[line]) / MCUS[mcu][line] * 100)
      print(f"SECTION: {line} size: {MCUS[mcu][line]} USED: {calcs[line]}({used_percent:.2f}%) FREE: {MCUS[mcu][line] - calcs[line]}")
    else:
      print(line, calcs[line])
  print()


if __name__ == "__main__":
  # red panda
  check_space("../board/obj/bootstub.panda_h7.elf", "H7")
  check_space("../board/obj/panda_h7.elf", "H7")
  # black panda
  check_space("../board/obj/bootstub.panda.elf", "F4")
  check_space("../board/obj/panda.elf", "F4")
  # jungle v1
  check_space("../board/jungle/obj/bootstub.panda_jungle.elf", "F4")
  check_space("../board/jungle/obj/panda_jungle.elf", "F4")
  # jungle v2
  check_space("../board/jungle/obj/bootstub.panda_jungle_h7.elf", "H7")
  check_space("../board/jungle/obj/panda_jungle_h7.elf", "H7")
