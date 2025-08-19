#!/usr/bin/env python3
import time
import re
from panda import PandaJungle

RED = '\033[91m'
GREEN = '\033[92m'

def colorize_errors(value):
  if isinstance(value, str):
    if re.search(r'(?i)No error', value):
      return f'{GREEN}{value}\033[0m'
    elif re.search(r'(?i)(?<!No error\s)(err|error)', value):
      return f'{RED}{value}\033[0m'
  return str(value)

if __name__ == "__main__":
  jungle = PandaJungle()

  while True:
    print(chr(27) + "[2J") # clear screen
    print("Connected to " + ("internal panda" if jungle.is_internal() else "External panda") + f" id: {jungle.get_serial()[0]}: {jungle.get_version()}")
    for bus in range(3):
      print(f"\nBus {bus}:")
      health = jungle.can_health(bus)
      for key, value in health.items():
        print(f"{key}: {colorize_errors(value)}  ", end=" ")
    print()
    time.sleep(1)
